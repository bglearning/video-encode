"""save embeddings."""
import os
import json

import fsspec
import numpy as np
import webdataset as wds

from io import BytesIO


class FileWriter:
    """Writes output as files."""

    def __init__(self, output_folder, output_key_start: int=0):
        self.output_folder = output_folder
        self.output_key_start = output_key_start

        self.fs, self.output_folder = fsspec.core.url_to_fs(output_folder)

    def write(self, arr, key, metadata=None):
        """write sample to file."""
        try:
            # key is probably pyarrow.lib.Int64Scalar
            # which can't be directly converted into a string
            key_int = int(str(key))
            key = key_int + self.output_key_start
        except (ValueError, TypeError):
            pass

        key = str(key)

        if metadata is not None and "numpy_metadata" in metadata:
            save_pth = os.path.join(self.output_folder, key + "-fea.npz")
            np_metadata = metadata.pop("numpy_metadata")
            np_metadata["clip_embeddings"] = arr

            with self.fs.open(save_pth, "wb") as f:
                nbp = BytesIO()
                np.savez(nbp, np_metadata)
                f.write(nbp.getbuffer())
        else:
            save_pth = os.path.join(self.output_folder, key + ".npy")
            with self.fs.open(save_pth, "wb") as f:
                nbp = BytesIO()
                np.save(nbp, arr)
                f.write(nbp.getbuffer())

        if metadata is not None:
            if "caption" in metadata:
                caption = str(metadata.pop("caption"))
                caption_filename = os.path.join(self.output_folder, key + ".txt")
                with self.fs.open(caption_filename, "w") as f:
                    f.write(caption)
            if len(metadata) > 0:
                if "mp4_video" in metadata:
                    vid_bytes = metadata.pop("mp4_video")
                    vid_filename = os.path.join(self.output_folder, key + ".mp4")
                    with self.fs.open(vid_filename, "w") as f:
                        f.write(vid_bytes)
                j = json.dumps(metadata, indent=4)
                meta_filename = os.path.join(self.output_folder, key + ".json")
                with self.fs.open(meta_filename, "w") as f:
                    f.write(j)

    def close(self):
        pass


class WebDatasetWriter:
    """Writes output in WebDataset format."""

    def __init__(self, output_folder, oom_shard_count, encode_format, maxcount=10000, shard_id=0):
        self.output_folder = output_folder
        self.oom_shard_count = oom_shard_count
        self.encode_format = encode_format
        self.maxcount = maxcount
        self.shard_id = shard_id
        self.shard_suffix = "clip_embeddings"  # TODO: maybe there should be param for this?

        self.count = 0

        self.tarwriter = None
        self.tar_fd = None

        self.create_shard()

    def create_shard(self, shard_id=None):
        """create new shard in sequential order."""
        self.close()
        if shard_id is not None:
            self.shard_id = shard_id
        shard_name = "{shard_id:0{oom_shard_count}d}".format(  # pylint: disable=consider-using-f-string
            shard_id=self.shard_id, oom_shard_count=self.oom_shard_count
        )
        shard_name += "_" + self.shard_suffix

        fs, output_path = fsspec.core.url_to_fs(self.output_folder)
        self.tar_fd = fs.open(f"{output_path}/{shard_name}.tar", "wb")
        self.tarwriter = wds.TarWriter(self.tar_fd)

    def write(self, arr, key, metadata=None):
        """write sample to current shard."""
        key = str(key)
        if self.count >= self.maxcount:
            self.shard_id += 1
            self.count = 0
            self.create_shard()

        sample = {"__key__": key}
        if metadata is not None and "numpy_metadata" in metadata:
            numpy_metadata = metadata.pop("numpy_metadata")
            # TODO: maybe add model type
            if arr is not None:
                numpy_metadata["clip_embeddings"] = arr
            sample["npz"] = numpy_metadata
        elif arr is not None:
            sample[self.encode_format] = arr
        if metadata is not None:
            if "caption" in metadata:
                sample["txt"] = str(metadata.pop("caption"))
            if len(metadata) > 0:
                if "mp4_video" in metadata:
                    vid_bytes = metadata.pop("mp4_video")
                    sample["mp4"] = vid_bytes
                sample["json"] = json.dumps(metadata, indent=4)

        self.tarwriter.write(sample)
        self.count += 1

    def close(self):
        if self.tarwriter is not None:
            self.tarwriter.close()
            self.tar_fd.close()
