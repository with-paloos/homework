from typing import Iterable, Callable
from itertools import tee, islice, zip_longest
from dataclasses import dataclass, field

MB = 1024 * 1024


def validate_positive_integer(value, name):
    if not isinstance(value, int):
        raise ValueError(f'Expected {name} to be an integer, got {type(value)}')
    if value <= 0:
        raise ValueError(f'Expected {name} to be positive, got {value}')


@dataclass
class Record:
    content: str
    size_bytes: int

    @staticmethod
    def string_size_bytes(inp: str) -> int:
        return len(inp.encode('utf-8'))

    @classmethod
    def from_string(cls, string: str) -> 'Record':
        return cls(
            string,
            Record.string_size_bytes(string)
        )


@dataclass
class Batch:
    records: list[Record] = field(default_factory=list)
    size_bytes: int = 0
    size_records: int = 0

    def add_record(self, record) -> 'Batch':
        """
        Add new record to batch immutably.
        """
        return Batch(
            self.records + [record],
            self.size_bytes + record.size_bytes,
            self.size_records + 1
        )

    def to_list(self) -> list[str]:
        return list(r.content for r in self.records)


def _batch_records(
    records: Iterable[Record],
    valid_batch: Callable[[Batch], bool],
    valid_record: Callable[[Record], bool]
    ) -> Iterable[Batch]:
    """
    Batch data records into batches with respecting constraint predicates.
    Processing is lazy, so it can be used with infinite streams.

    Args:
        records: Iterable of Record instances.
        valid_batch: Predicate to check if a batch is valid.
        valid_record: Predicate to check if a record is valid.
    
    Returns:
        Iterable of Batch instances.
    """

    #  Filter only valid records
    xs = filter(valid_record, records)

    # Setup lazy lookahead
    (xs, ys) = tee(xs, 2)  # Duplicate iterable xs
    ys = islice(ys, 1, None)  # Skip first item, now ys form a lookahead in respect of xs

    b = Batch()
    for x, lookahead in zip_longest(xs, ys):
        b = b.add_record(x)
        # If lookahead is None we are reached end of data
        if lookahead is None or not valid_batch(b.add_record(lookahead)):
            yield b
            b = Batch()


def batch_records(
    data: Iterable[str],
    max_record_size: int = 1 * MB,
    max_batch_size: int = 5 * MB,
    max_records_per_batch: int = 500
    ) -> list[list[str]]:
    """
    Batch data records with size less than max_record_size into batches, where each batch
    contains at most max_records_per_batch records and the total size of the batch is less
    than max_batch_size.

    Args:
        data: Iterable of strings, where each string represents a record.
        max_record_size: Maximum size of a record in bytes. Default is 1MB.
        max_batch_size: Maximum size of a batch in bytes. Default is 5MB.
        max_records_per_batch: Maximum number of records in a batch. Default is 500.

    Returns:
        List of batches, where each batch is a list of records.
    """
  
    validate_positive_integer(max_record_size, 'max_record_size')
    validate_positive_integer(max_batch_size, 'max_batch_size')
    validate_positive_integer(max_records_per_batch, 'max_records_per_batch')

    # If max_batch_size is smaller than max_record_size, use max_batch_size as filtering criteria
    # this prevents situation of having unbatchable records after filtering valid records
    max_record_size = min(max_record_size, max_batch_size)

    def valid_batch(batch):
        return batch.size_bytes <= max_batch_size and batch.size_records <= max_records_per_batch

    
    def valid_record(record):
        return record.size_bytes <= max_record_size

    # Convert data into Record instances
    xs = map(Record.from_string, data)

    # Generate batches
    ys = _batch_records(xs, valid_batch, valid_record)
    
    # Convert batches into lists of strings
    return list(map(lambda y: y.to_list(), ys))
