"""encode video with CLIP"""
import sys
import time

# import open_clip
import math
import numpy as np
import torch

from torchvision.transforms import ToPILImage
from video2numpy.frame_reader import FrameReader

from .reader import Reader
# from .simplemapper import FrameMapper
from .encoders import create_encoder, VideoEncoder
from .utils import block2dl
from .writer import FileWriter, WebDatasetWriter
from .distributed import world_info_from_env

import tarfile
import tempfile
import os
import json
import braceexpand
import glob
import fsspec
import io

from .blip_utils import blip_model, blip_text_processors, blip_vis_processors

BATCH_SIZE = 384
IMG_SIZE = 224
EMB_DIM = 512
N_DATASET_WORKERS = 4
CHUNK_SIZE = 200


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def encode_chunk(
    frames,
    ind_dict,
    writer,
    mapper: VideoEncoder,
    preprocess,
    meta,
    ids,
    use_dst_name,
    device,
    input_format="table",
    captioning_strategy="none",
):
    """encodes a chunk of video frames and saves."""
    vid_block = np.concatenate(frames)
    dl = block2dl(vid_block, preprocess, BATCH_SIZE, N_DATASET_WORKERS)

    if captioning_strategy == "none":
        embeddings = []
        for batch in dl:
            with torch.no_grad(), torch.cuda.amp.autocast():
                emb = mapper(batch.to(device))
                embeddings.append(emb)

        """TODO Reintroduce this later
        caption_embs = None
        if mapper.txt_processor is not None:
            # TODO: is there a better way of doing this?
            # here we will compute similarity of empty string...
            captions = [m["caption"] if "caption" in m else "" for m in meta]
            caption_embs = mapper.encode_captions(captions)
            caption_embs = caption_embs / np.linalg.norm(caption_embs, axis=-1)[:, None]
        """

        embeddings = np.concatenate(embeddings)
        for ref, (i0, it, dst_name) in ind_dict.items():
            vid_id = dst_name[:-4] if use_dst_name else ids[ref]
            if input_format == "webdataset":
                vid_meta = meta[ref]
            else:
                vid_meta = {}
                for k in meta:
                    vid_meta[k] = meta[k][ref].as_py()

            frame_embeddings = embeddings[i0:it]
            # TODO Reintroduce this later
            """
            if caption_embs is not None:
                # normalize
                fe = frame_embeddings / np.linalg.norm(frame_embeddings, axis=-1)[:, None]
                ce = caption_embs[ref]

                sim = (fe @ ce.T).tolist()

                vid_meta["clip_frame_similarity"] = sim
            """

            writer.write(frame_embeddings, vid_id, vid_meta)
    else:
        captions = []
        for batch in dl:
            captions += mapper.generate_captions(batch.to(device))

        for ref, (i0, it, dst_name) in ind_dict.items():
            vid_id = dst_name[:-4] if use_dst_name else ids[ref]
            if input_format == "webdataset":
                vid_meta = meta[ref]
            else:
                vid_meta = {}
                for k in meta:
                    vid_meta[k] = meta[k][ref].as_py()

            vid_meta["generated_caption"] = captions[i0:it]
            # TODO: we should be able to do both at once with a CoCa model
            writer.write(None, vid_id, vid_meta)


def read_shard(tempdir, pass_through_keys=None):
    """
    Extract video filepaths, video ids, and metadata from the contents of an opened WebDataset shard

    Input:
        tempdir:
            path to directory containing contents of an opened WebDataset shard with input data
    """
    if pass_through_keys is None:
        pass_through_keys = []

    vids = sorted(
        [f.split("/")[-1] for f in glob.glob(tempdir + "/" + "*.mp4")]
    )  # TODO: parameterize the video extension

    has_npz = len(glob.glob(tempdir + "/" + "*.npz")) > 0
    has_txt = len(glob.glob(tempdir + "/" + "*.txt")) > 0
    has_json = len(glob.glob(tempdir + "/" + "*.json")) > 0

    keys = [x.split(".mp4")[0] for x in vids]
    meta = []
    for key in keys:
        if has_json and "json" in pass_through_keys:
            with open(tempdir + "/" + key + ".json", "rb") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        if has_txt and "txt" in pass_through_keys:
            with open(tempdir + "/" + key + ".txt", "r", encoding="UTF-8") as f:
                txt = f.read()
            metadata["caption"] = txt

        if "mp4" in pass_through_keys:
            with open(tempdir + "/" + key + ".mp4", "rb") as f:
                mp4_video = f.read()
                metadata["mp4_video"] = mp4_video
        if has_npz and "npz" in pass_through_keys:
            np_metadata = dict(np.load(tempdir + "/" + key + ".npz"))
            metadata["numpy_metadata"] = np_metadata

        meta.append(metadata)

    vids = [tempdir + "/" + v for v in vids]
    return vids, keys, meta


def clip_video_encode(
    src,
    dest="",
    output_format="files",
    take_every_nth=25,
    encoder_type="open_clip",
    input_format="table",
    frame_workers=1,
    frame_memory_size=4,
    metadata_columns="",
    use_dst_name=False,
    distribute="none",
    oom_shard_count=5,
    output_key_start=0,
    oc_model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    captioning_strategy="none",
    pass_through_keys="mp4,txt,json",
    caption_similarity=False,
    n_final_frames=None,
):
    """
    Encode frames using CLIP image encoder

    Input:
      src:
        str: path to mp4 file
        str: youtube link
        str: path to txt file with multiple mp4's or youtube links
        list: list with multiple mp4's or youtube links
      dest:
        str: directory where to save embeddings to
        None: dest = src + .npy
      output_format:
        str: "files" or "webdataset"
      take_every_nth:
        int: only take every nth frame
      encoder_type:
        str: Encoder to use among ('open_clip', 'openai_clip', 'blip2')
      frame_workers:
        int: number of Processes to distribute video reading to.
      frame_memory_size:
        int: GB of memory for FrameReader.
      metadata_columns:
        str: a comma separated list of metadata column names to look for in src
      use_dst_name:
        bool: use the save name suggested by video2numpy
      distribute:
        str: distribution strategy, currently either slurm or none
      oc_model_name:
        str: open_clip model name, used for selecting CLIP architecture
      pretrained:
        str: open_clip pretrained weights name
      captioning_strategy:
        str: which frames of a video to generate captions for. Possible values are:
          - none: don't generate any captions
          - center: generate a caption for the middle frame
        int: (NOT IMPLEMENTED) step size for which frames to generate captions for
      pass_through_keys:
        str: comma separated list of extension to pass through from input dataset (if webdataset format)
      caption_similarity:
        bool: whether to put the similarity between the average frame embedding and text embedding into metadata
    """
    assert input_format in ["table", "webdataset"]

    if isinstance(metadata_columns, str):
        metadata_columns = [metadata_columns] if metadata_columns != "" else []
    metadata_columns = list(metadata_columns) if isinstance(metadata_columns, tuple) else metadata_columns

    if isinstance(pass_through_keys, str):
        pass_through_keys = pass_through_keys.split(",")

    if input_format == "table":
        reader = Reader(src, metadata_columns)
        vids, ids, meta = reader.get_data()
        meta_refs = list(range(len(vids)))

    else:  # WebDataset, so we distribute shards
        shards = list(braceexpand.braceexpand(src))

    starting_shard_id = 0
    shard_sample_count = 10000

    if distribute == "slurm":
        local_rank, global_rank, world_size = world_info_from_env()
        if input_format == "table":
            work_size = math.ceil(len(vids) / world_size)
        else:
            work_size = math.ceil(len(shards) / world_size)
        print(f"Slurm worker {global_rank} processing {work_size} videos...")
        ws, wf = global_rank * work_size, (global_rank + 1) * work_size
        if input_format == "table":
            vids = vids[ws:wf]
            ids = ids[ws:wf]
            for mc in meta.keys():
                meta[mc] = meta[mc][ws:wf]

            starting_shard_id += math.ceil(work_size / shard_sample_count) * global_rank
        elif input_format == "webdataset":
            shards = shards[ws:wf]
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    else:
        local_rank, global_rank, world_size = 0, 0, 1  # TODO: how do we do this?
        device = "cuda" if torch.cuda.is_available() else "cpu"

    assert output_format in ["files", "webdataset"]
    if output_format == "files":
        writer = FileWriter(dest, output_key_start=output_key_start)
    elif output_format == "webdataset":
        # TODO: maybe include params for this?
        starting_shard_id = int(shards[0].split("/")[-1].split(".tar")[0])
        writer = WebDatasetWriter(dest, oom_shard_count, "npy", maxcount=1e6, shard_id=starting_shard_id)

    # Initialize model:
    fm = create_encoder(encoder_type=encoder_type)
    preprocess = fm.preprocessor

    if input_format == "table":
        fr = FrameReader(
            vids,
            meta_refs,
            take_every_nth,
            IMG_SIZE,
            workers=frame_workers,
            memory_size=frame_memory_size,
        )
        fr.start_reading()

        frames, ind_dict = [], {}
        block_size = 0
        i = 0
        for vid_frames, info in fr:
            i += 1
            if n_final_frames is not None:
                # (Uniformly) Sample n_final_frames instead of taking all frames from the reader
                # Shape of vid_frames is (n_frames, 224, 224, 3)
                sample_idx = np.round(np.linspace(0, len(vid_frames) - 1, n_final_frames)).astype(int)
                vid_frames = vid_frames[sample_idx]

            frames.append(vid_frames)
            ind_dict[info["reference"]] = (
                block_size,
                block_size + vid_frames.shape[0],
                info["dst_name"],
            )
            block_size += vid_frames.shape[0]

            if i % CHUNK_SIZE == 0:
                encode_chunk(
                    frames, ind_dict, writer, fm, preprocess, meta, ids, use_dst_name, device, input_format=input_format
                )
                frames, ind_dict, block_size = [], {}, 0

        if len(frames) > 0:  # TODO: make this cleaner
            encode_chunk(
                frames, ind_dict, writer, fm, preprocess, meta, ids, use_dst_name, device, input_format=input_format
            )
    else:  # WebDataset shard logic
        for shard in shards:
            times = {}
            t = time.time()
            with tempfile.TemporaryDirectory(prefix=f"worker_{global_rank}_") as tempdir:
                os.chmod(tempdir, 0o777)  # This lets subprocesses from v2np read files in the tempdir
                folder = "/".join(shard.split("/")[0:-1])
                fs, output_path = fsspec.core.url_to_fs(folder)

                shard_id = shard.split("/")[-1]
                tar_bytes = io.BytesIO(fs.open(f"{output_path}/{shard_id}").read())
                with tarfile.open(fileobj=tar_bytes) as tar:
                    tar.extractall(tempdir)
                writer.create_shard(shard_id=int(shard_id.split(".tar")[0]))
                times["download_and_extract"] = times.get("download_and_extract", 0) + time.time() - t
                t = time.time()
                vids, ids, meta = read_shard(tempdir, pass_through_keys=pass_through_keys)
                meta_refs = list(range(len(vids)))
                fr = FrameReader(
                    vids,
                    meta_refs,
                    take_every_nth,
                    IMG_SIZE,
                    workers=frame_workers,
                    memory_size=frame_memory_size,
                )
                fr.start_reading()

                frames, ind_dict = [], {}
                block_size = 0
                i = 0
                n_frames = 0
                for vid_frames, info in fr:
                    i += 1

                    if captioning_strategy == "center":
                        vid_frames = vid_frames[len(vid_frames) // 2 : len(vid_frames) // 2 + 1]

                    if n_final_frames is not None:
                        sample_idx = np.round(np.linspace(0, len(vid_frames) - 1, n_final_frames)).astype(int)
                        vid_frames = vid_frames[sample_idx]

                    n_frames += len(vid_frames)
                    frames.append(vid_frames)
                    ind_dict[info["reference"]] = (
                        block_size,
                        block_size + vid_frames.shape[0],
                        info["dst_name"],
                    )
                    block_size += vid_frames.shape[0]
                    times["read_frames"] = times.get("read_frames", 0) + time.time() - t
                    t = time.time()

                    if i % CHUNK_SIZE == 0:
                        encode_chunk(
                            frames,
                            ind_dict,
                            writer,
                            fm,
                            preprocess,
                            meta,
                            ids,
                            use_dst_name,
                            device,
                            input_format=input_format,
                            captioning_strategy=captioning_strategy,
                        )
                        times["encode"] = times.get("encode", 0) + time.time() - t
                        t = time.time()
                        frames, ind_dict, block_size = [], {}, 0
                t = time.time()
                if len(frames) > 0:  # TODO: make this cleaner
                    encode_chunk(
                        frames,
                        ind_dict,
                        writer,
                        fm,
                        preprocess,
                        meta,
                        ids,
                        use_dst_name,
                        device,
                        input_format=input_format,
                        captioning_strategy=captioning_strategy,
                    )
                times["encode"] = times.get("encode", 0) + time.time() - t
                t = time.time()
            frame_adjusted = {k: n_frames / v for k, v in times.items()}
            print(f"Frames/s: {frame_adjusted}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python clip-video-encode.py video.mp4 embeddings.npy take_every_nth")
        sys.exit(1)
    clip_video_encode(sys.argv[1], sys.argv[2], int(sys.argv[3]))
