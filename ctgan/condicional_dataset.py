"""CondicionalDataset — Dataset PyTorch com suporte a amostragem condicional.

Elimina a necessidade de conversão NumPy → Tensor a cada batch,
movendo a conversão para antes do loop de treinamento.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class CondicionalDataset(Dataset):
    """Dataset que mantém dados convertidos em tensores PyTorch.

    Converte o dataset inteiro uma única vez, eliminando conversões
    repetidas NumPy → Tensor durante o treinamento.

    A conversão inicial compartilha memória com o array NumPy de origem
    (``torch.from_numpy``), portanto não há cópia extra quando os dados já
    são ``float32``. A referência NumPy é liberada após ``to()`` sempre que
    os dados forem efetivamente copiados (pin/transferência para device).
    """

    def __init__(self, data_numpy, device='cpu', pin_memory=False, keep_on_cpu=False):
        """Converte o dataset inteiro para tensor uma única vez.

        Args:
            data_numpy: array NumPy de forma (n_rows, n_features)
            device: device PyTorch de destino ('cpu' ou 'cuda')
            pin_memory: se True, usa memória paginada (pinned) na CPU para
                transferências assíncronas por batch
            keep_on_cpu: se True, mantém os dados na CPU mesmo com ``device``
                em GPU, transferindo apenas cada batch sob demanda (útil para
                tabelas maiores que a VRAM)
        """
        if data_numpy.dtype != np.float32:
            data_numpy = data_numpy.astype('float32')
        if not data_numpy.flags['C_CONTIGUOUS']:
            data_numpy = np.ascontiguousarray(data_numpy)

        self._data_numpy = data_numpy
        # Compartilha o buffer NumPy — nenhuma cópia adicional.
        self.data = torch.from_numpy(data_numpy)

        self.n_rows = self.data.shape[0]
        self.n_features = self.data.shape[1]
        self._target_device = torch.device(device)
        self._keep_on_cpu = keep_on_cpu
        self._pin_memory = pin_memory

    @property
    def data_numpy(self):
        """Array NumPy de origem (disponível até os dados serem copiados)."""
        if self._data_numpy is None:
            raise RuntimeError(
                'Os dados NumPy foram liberados após mover para o device. '
                'Construa o CondicionalSampler antes de chamar .to().'
            )
        return self._data_numpy

    def to(self, device=None):
        """Move os dados para o device e libera a referência NumPy da CPU.

        Deve ser chamado somente depois que o sampler já construiu seus
        índices a partir de ``data_numpy``.
        """
        if device is not None:
            self._target_device = torch.device(device)

        copied = False

        if self._pin_memory and not self.data.is_pinned():
            self.data = self.data.pin_memory()
            copied = True

        if not self._keep_on_cpu and self._target_device.type != 'cpu':
            self.data = self.data.to(self._target_device)
            copied = True

        if copied:
            # Libera a referência NumPy (a de CPU ainda viva).
            self._data_numpy = None

        return self

    def __len__(self):
        return self.n_rows

    def __getitem__(self, idx):
        batch = self.data[idx]
        if batch.device != self._target_device:
            batch = batch.to(self._target_device, non_blocking=self.data.is_pinned())
        return batch


class CondicionalSampler:
    """Sampler que realiza amostragem condicional sobre tensores PyTorch.

    Mantém a lógica de amostragem condicional do CTGAN original,
    mas opera diretamente sobre tensores já convertidos.

    Os índices por categoria são armazenados em uma estrutura CSR única
    (``indices int32`` + ``offsets``), construída a partir do array NumPy
    original — sem round-trip GPU→CPU.
    """

    def __init__(self, dataset, output_info, log_frequency):
        """Inicializa o sampler com os dados já convertidos.

        Args:
            dataset: CondicionalDataset com dados em tensores
            output_info: informação de saída do DataTransformer
            log_frequency: se True, usa frequência logarítmica
        """
        self._dataset = dataset
        self._data_length = len(dataset)

        def is_discrete_column(column_info):
            return len(column_info) == 1 and column_info[0].activation_fn == 'softmax'

        n_discrete_columns = sum([
            1 for column_info in output_info if is_discrete_column(column_info)
        ])

        self._discrete_column_matrix_st = np.zeros(n_discrete_columns, dtype='int32')

        # C6: usa o array NumPy original em vez de trazê-lo de volta da GPU.
        data_numpy = dataset.data_numpy

        # Preparar matriz de probabilidades
        max_category = max(
            [column_info[0].dim for column_info in output_info if is_discrete_column(column_info)],
            default=0,
        )

        self._discrete_column_cond_st = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_n_category = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_category_prob = np.zeros((n_discrete_columns, max_category))
        self._n_discrete_columns = n_discrete_columns
        self._n_categories = sum([
            column_info[0].dim for column_info in output_info if is_discrete_column(column_info)
        ])

        # C7: estrutura CSR única — indices int32 + offsets por categoria.
        csr_indices = []
        csr_offsets = np.zeros(self._n_categories + 1, dtype='int64')
        global_category_id = 0

        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim
                block = data_numpy[:, st:ed]

                # Código da categoria ativa em cada linha (one-hot).
                codes = np.argmax(block, axis=1)
                present = block[np.arange(block.shape[0]), codes] > 0
                rows = np.flatnonzero(present)

                # argsort do código agrupa as linhas por categoria de uma vez,
                # substituindo o np.nonzero por coluna do código original.
                order = np.argsort(codes[rows], kind='stable')
                sorted_rows = rows[order].astype('int32')
                counts = np.bincount(codes[rows], minlength=span_info.dim)

                csr_indices.append(sorted_rows)
                csr_offsets[global_category_id + 1: global_category_id + span_info.dim + 1] = (
                    csr_offsets[global_category_id] + np.cumsum(counts)
                )
                global_category_id += span_info.dim

                category_freq = np.sum(block, axis=0)
                if log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                self._discrete_column_category_prob[current_id, : span_info.dim] = category_prob
                self._discrete_column_cond_st[current_id] = current_cond_st
                self._discrete_column_n_category[current_id] = span_info.dim
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])

        if csr_indices:
            self._csr_indices = np.concatenate(csr_indices)
        else:
            self._csr_indices = np.zeros(0, dtype='int32')
        self._csr_offsets = csr_offsets

    def _random_choice_prob_index(self, discrete_column_id):
        probs = self._discrete_column_category_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)

    def sample_condvec(self, batch):
        """Gera o vetor condicional para treinamento.

        Returns:
            cond (batch x #categories):
                The conditional vector.
            mask (batch x #discrete columns):
                A one-hot vector indicating the selected discrete column.
            discrete column id (batch):
                Integer representation of mask.
            category_id_in_col (batch):
                Selected category in the selected discrete column.
        """
        if self._n_discrete_columns == 0:
            return None

        discrete_column_id = np.random.choice(
            np.arange(self._n_discrete_columns), batch
        )

        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')
        mask[np.arange(batch), discrete_column_id] = 1
        category_id_in_col = self._random_choice_prob_index(discrete_column_id)
        category_id = self._discrete_column_cond_st[discrete_column_id] + category_id_in_col
        cond[np.arange(batch), category_id] = 1

        return cond, mask, discrete_column_id, category_id_in_col

    def sample_original_condvec(self, batch):
        """Gera o vetor condicional para geração usando frequência original."""
        if self._n_discrete_columns == 0:
            return None

        category_freq = self._discrete_column_category_prob.flatten()
        category_freq = category_freq[category_freq != 0]
        category_freq = category_freq / np.sum(category_freq)
        col_idxs = np.random.choice(np.arange(len(category_freq)), batch, p=category_freq)
        cond = np.zeros((batch, self._n_categories), dtype='float32')
        cond[np.arange(batch), col_idxs] = 1

        return cond

    def sample_data(self, n, col, opt):
        """Amostra dados diretamente do tensor PyTorch.

        Args:
            n: número de linhas para amostrar
            col: colunas discretas para condicionar
            opt: valores das colunas

        Returns:
            tensor PyTorch já no device correto
        """
        if col is None:
            idx = np.random.randint(self._data_length, size=n)
            return self._dataset[torch.tensor(idx, dtype=torch.long)]

        # C8: sorteio vetorizado de offsets dentro dos ranges CSR.
        col = np.asarray(col, dtype='int64')
        opt = np.asarray(opt, dtype='int64')
        global_cat = self._discrete_column_cond_st[col] + opt
        starts = self._csr_offsets[global_cat]
        ends = self._csr_offsets[global_cat + 1]

        spans = ends - starts
        # Ranges vazios nunca são sorteados pelo condvec, mas evitamos
        # estourar o array caso ocorram (span 0 → posição fixada em starts).
        positions = starts + (np.random.rand(col.shape[0]) * np.maximum(spans, 1)).astype('int64')
        np.clip(positions, 0, self._csr_indices.shape[0] - 1, out=positions)

        idx = self._csr_indices[positions]
        idx_tensor = torch.tensor(idx.astype('int64'), dtype=torch.long)
        return self._dataset[idx_tensor]

    def dim_cond_vec(self):
        """Return the total number of categories."""
        return self._n_categories

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        """Generate the condition vector."""
        vec = np.zeros((batch, self._n_categories), dtype='float32')
        id_ = self._discrete_column_matrix_st[condition_info['discrete_column_id']]
        id_ += condition_info['value_id']
        vec[:, id_] = 1
        return vec
