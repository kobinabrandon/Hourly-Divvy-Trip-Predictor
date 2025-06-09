import pandas as pd 


class CutoffIndexer:
    def __init__(self, ts_data: pd.DataFrame, input_seq_len: int, step_size: int) -> None:
        """
        Allows us to invoke a particular method of getting the cutoff indices for each 
        station. These indices will be needed when converting time series data into 
        training data.

        Args:
            ts_data (pd.DataFrame): the time series dataset that serves as the input
            input_seq_len (int): the number of rows to be considered at any one time
            step_size (int): how many rows down we move as we repeat the process
        """
        self.step_size: int = step_size
        self.ts_data: pd.DataFrame = ts_data
        self.input_seq_len: int = input_seq_len
        self.stop_position: int = len(ts_data) - 1

        self.indices = self.get_cutoff_indices()
        self.use_standard_indexer: bool = self.use_standard_cutoff_indexer()
        self.indices: list[tuple[int, int, int]] = self.get_cutoff_indices()

    def use_standard_cutoff_indexer(self) -> bool:
        """
        Determines whether the standard cutoff indexer is to be used, based on the number of rows 
        in the time series data. In particular, the function checks whether the input sequence
        length is no more than the length of the data. This condition is required for the standard
        indexer to be used in the first place.

        Returns:
            bool: whether to use the standard indexer or not.
        """
        stop_position = len(self.ts_data) - 1  
        return True if stop_position >= self.input_seq_len + 1 else False

    def get_cutoff_indices(self) -> list[tuple[int, int, int]]:
        """
        Returns:
            list: the list of cutoff indices
        """
        if self.use_standard_indexer:
            indices = self.run_standard_cutoff_indexer(
                first_index=0, 
                mid_index=self.input_seq_len, 
                last_index=self.input_seq_len+1
            )     

            return indices
            
        elif not self.use_standard_indexer and len(self.ts_data) >= 2:
            indices = self.run_modified_cutoff_indexer(first_index=0, mid_index=1, last_index=2)
            return indices

        elif not self.use_standard_indexer and len(self.ts_data) == 1:
            return [self.ts_data.index[0]]

    def run_modified_cutoff_indexer(self, first_index: int, mid_index: int, last_index: int) -> list[tuple[int, int, int]]:
        """
        A modified version of the standard indexer, which is meant to deal with a specific problem that emerges when
        the given station's time series data has only two rows.

        Args:
            first_index:
            mid_index:
            last_index:

        Returns:
        """
        indices: list[tuple[int, int, int]] = []
        while mid_index <= self.stop_position:
            index = (first_index, mid_index, last_index)
            indices.append(index)
        
            first_index += self.step_size
            mid_index += self.step_size
            last_index += self.step_size

        return indices

    def run_standard_cutoff_indexer(self, first_index: int, mid_index: int, last_index: int) -> list[tuple[int, int, int]]:
        """
        Starts by taking a certain number of rows of a given dataframe as an input, and the
        indices of the row on which the selected rows start and end. These will be placed
        in the first and second positions of a three element tuple. The third position of
        said tuple will be occupied by the index of the row that comes after.

        Then the function will slide "step_size" steps and repeat the process. The function
        terminates once it reaches the last row of the dataframe. 

        Credit to Pau. 

        Args:
            first_index (int): _description_
            mid_index (int): _description_
            last_index (int): _description

        Returns:
            list[tuple[int]]: _description_
        """
        indices: list[tuple[int, int, int]] = []
        while last_index <= self.stop_position: 
            index = (first_index, mid_index, last_index)
            indices.append(index)

            first_index += self.step_size
            mid_index += self.step_size
            last_index += self.step_size
            
        return indices

