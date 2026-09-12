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
    """

    def __init__(self, data_numpy, device='cpu'):
        """Converte o dataset inteiro para tensor uma única vez.

        Args:
            data_numpy: array NumPy de forma (n_rows, n_features)
            device: device PyTorch ('cpu' ou 'cuda')
        """
        self.data = torch.from_numpy(
            data_numpy.astype('float32')
        ).to(device)
        self.n_rows = self.data.shape[0]

    def __len__(self):
        return self.n_rows

    def __getitem__(self, idx):
        return self.data[idx]


class CondicionalSampler:
    """Sampler que realiza amostragem condicional sobre tensores PyTorch.

    Mantém a lógica de amostragem condicional do CTGAN original,
    mas opera diretamente sobre tensores já convertidos.
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
        self._rid_by_cat_cols = []

        # Converter tensor para NumPy para indexação (necessário para np.nonzero)
        data_numpy = dataset.data.cpu().numpy()

        st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim

                rid_by_cat = []
                for j in range(span_info.dim):
                    rid_by_cat.append(np.nonzero(data_numpy[:, st + j])[0])
                self._rid_by_cat_cols.append(rid_by_cat)
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])

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

        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim
                category_freq = np.sum(data_numpy[:, st:ed], axis=0)
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

        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self._rid_by_cat_cols[c][o]))

        idx_tensor = torch.tensor(idx, dtype=torch.long)
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
