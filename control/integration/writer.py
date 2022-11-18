# 2022: Ildar Nurgaliev (i.nurgaliev@docet.ai)

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import List

import aiofiles
import pandas as pd
from ray.air._internal.json import SafeFallbackEncoder
from ray.rllib.offline.io_context import IOContext
from ray.rllib.offline.json_writer import _to_json_dict
from ray.rllib.offline.output_writer import OutputWriter
from ray.rllib.utils.typing import SampleBatchType

logger = logging.getLogger(__name__)


def _to_json(batch: SampleBatchType, compress_columns: List[str]) -> str:
    out = _to_json_dict(batch, compress_columns)
    del out['type']
    return json.dumps(out, cls=SafeFallbackEncoder)


async def _write(json_lines: List[str], fpath: str) -> None:
    async with aiofiles.open(fpath, mode='a+', encoding='utf-8') as f:
        for json_line in json_lines:
            await f.write(json_line)
            await f.write('\n')
        await f.flush()


class AsyncJsonWriter(OutputWriter):
    COUNT_TO_FLUSH = 2

    def __init__(self, store_path: str,
                 ioctx: IOContext = None,
                 max_file_size: int = 64 * 1024 * 1024,
                 compress_columns: List[str] = frozenset(["obs", "new_obs"]),
                 avoid_columns=frozenset(['obs', 'new_obs']),
                 prefix=None):
        self.ioctx = ioctx or IOContext()
        self.compress_columns = compress_columns
        self.avoid_columns = avoid_columns
        self.max_file_size = max_file_size

        store_path = os.path.abspath(os.path.expanduser(store_path))
        os.makedirs(store_path, exist_ok=True)
        assert os.path.exists(store_path), "Failed to create {}".format(store_path)
        self.store_path = store_path
        self.file_index = 0
        self._fpath = None
        self._buffer: List[str] = []
        self._prefix = prefix if prefix else 'output'
        self.bytes_written = 0

    def write(self, sample_batch: SampleBatchType, extra_info = None):
        # Make data column-wise
        sample_columnwise_dict = {k: v for k, v in sample_batch.items() if k not in self.avoid_columns}
        if 'infos' in sample_batch:
            infos = sample_batch['infos']
            df = pd.DataFrame.from_records(infos)
            info_d = df.to_dict(orient='list')
            sample_columnwise_dict.update(info_d)

        if extra_info:
            sample_columnwise_dict.update(extra_info)

        # noinspection PyTypeChecker
        json_line: str = _to_json(sample_columnwise_dict, self.compress_columns)
        self.bytes_written += len(json_line)

        self._buffer.append(json_line)
        if len(self._buffer) > self.COUNT_TO_FLUSH:
            asyncio.run(_write(self._buffer, fpath=self._get_fpath()))
            self._buffer = []

    def _get_fpath(self) -> str:
        if not self._fpath or self.bytes_written >= self.max_file_size:
            timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
            self._fpath = os.path.join(
                self.store_path,
                "{}_f-{}_worker-{}_{}.jsonl".format(
                    self._prefix,
                    self.file_index,
                    self.ioctx.worker_index,
                    timestr
                ),
            )
            self.file_index += 1
            self.bytes_written = 0
            logger.info("Writing to new output file {}".format(self._fpath))
        return self._fpath

    def close(self) -> None:
        # flush
        if len(self._buffer) > 0:
            asyncio.run(_write(self._buffer, fpath=self._get_fpath()))
            self._buffer = []
        self._fpath = None
