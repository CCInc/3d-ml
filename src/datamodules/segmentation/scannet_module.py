import csv
import json
import logging
import os
import tempfile
from typing import Any, Dict, Optional
from urllib.request import urlopen

import numpy as np
import torch
from plyfile import PlyData
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.datamodules.base_dataloader import Base3dDataModule
from src.datamodules.common import DataModuleConfig, DataModuleTransforms

log = logging.getLogger(__name__)


def get_release_scans(release_file):
    scan_lines = urlopen(release_file)
    scans = []
    for scan_line in scan_lines:
        scan_id = scan_line.decode("utf8").rstrip("\n")
        scans.append(scan_id)
    return scans


def represents_int(s):
    """if string s represents an int."""
    try:
        int(s)
        return True
    except ValueError:
        return False


def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isfile(out_file):
        print("\t" + url + " > " + out_file)
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, "w")
        f.close()
        urllib.request.urlretrieve(url, out_file_tmp)
        # urllib.urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)
    else:
        pass


def download_label_map(out_dir):
    log.info("Downloading ScanNet " + ScanNetConfig.RELEASE_NAME + " label mapping file...")
    files = [ScanNetConfig.LABEL_MAP_FILE]
    for file in files:
        url = ScanNetConfig.BASE_URL + ScanNetConfig.RELEASE_TASKS + "/" + file
        localpath = os.path.join(out_dir, file)
        localdir = os.path.dirname(localpath)
        if not os.path.isdir(localdir):
            os.makedirs(localdir)
        download_file(url, localpath)
    log.info("Downloaded ScanNet " + ScanNetConfig.RELEASE_NAME + " label mapping file.")


def download_scan(scan_id, out_dir, file_types, use_v1_sens):
    # print("Downloading ScanNet " + RELEASE_NAME + " scan " + scan_id + " ...")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for ft in file_types:
        v1_sens = use_v1_sens and ft == ".sens"
        url = (
            ScanNetConfig.BASE_URL + ScanNetConfig.RELEASE + "/" + scan_id + "/" + scan_id + ft
            if not v1_sens
            else ScanNetConfig.BASE_URL
            + ScanNetConfig.RELEASES[ScanNetConfig.V1_IDX]
            + "/"
            + scan_id
            + "/"
            + scan_id
            + ft
        )
        out_file = out_dir + "/" + scan_id + ft
        download_file(url, out_file)
    print("Downloaded scan " + scan_id)


def download_release(release_scans, out_dir, file_types, use_v1_sens):
    if len(release_scans) == 0:
        return
    print("Downloading ScanNet " + ScanNetConfig.RELEASE_NAME + " release to " + out_dir + "...")
    failed = []
    for scan_id in release_scans:
        scan_out_dir = os.path.join(out_dir, scan_id)
        try:
            download_scan(scan_id, scan_out_dir, file_types, use_v1_sens)
        except:
            failed.append(scan_id)
    print("Downloaded ScanNet " + ScanNetConfig.RELEASE_NAME + " release.")
    if len(failed):
        log.warning(f"Failed downloads: {failed}")


def read_mesh_vertices_rgb(filename):
    """read XYZ RGB for each vertex.

    Note: RGB values are in 0-255
    """
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
        vertices[:, 3] = plydata["vertex"].data["red"]
        vertices[:, 4] = plydata["vertex"].data["green"]
        vertices[:, 5] = plydata["vertex"].data["blue"]
    return vertices


def read_label_mapping(filename, label_from="raw_category", label_to="nyu40id"):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping


def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data["segGroups"])
        for i in range(num_objects):
            object_id = data["segGroups"][i]["objectId"] + 1  # instance ids should be 1-indexed
            label = data["segGroups"][i]["label"]
            segs = data["segGroups"][i]["segments"]
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data["segIndices"])
        for i in range(num_verts):
            seg_id = data["segIndices"][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def export(mesh_file, agg_file, seg_file, meta_file, label_map_file, output_file=None):
    """points are XYZ RGB (RGB in 0-255), semantic label as nyu40 ids, instance label as
    1-#instance, box as (cx,cy,cz,dx,dy,dz,semantic_label)"""
    label_map = read_label_mapping(label_map_file, label_from="raw_category", label_to="nyu40id")
    mesh_vertices = read_mesh_vertices_rgb(mesh_file)

    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    for line in lines:
        if "axisAlignment" in line:
            axis_align_matrix = [
                float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")
            ]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    mesh_vertices[:, 0:3] = pts[:, 0:3]

    # Load semantic and instance labels
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]
    instance_bboxes = np.zeros((num_instances, 7))
    for obj_id in object_id_to_segs:
        label_id = object_id_to_label_id[obj_id]
        obj_pc = mesh_vertices[instance_ids == obj_id, 0:3]
        if len(obj_pc) == 0:
            continue
        # Compute axis aligned box
        # An axis aligned bounding box is parameterized by
        # (cx,cy,cz) and (dx,dy,dz) and label id
        # where (cx,cy,cz) is the center point of the box,
        # dx is the x-axis length of the box.
        xmin = np.min(obj_pc[:, 0])
        ymin = np.min(obj_pc[:, 1])
        zmin = np.min(obj_pc[:, 2])
        xmax = np.max(obj_pc[:, 0])
        ymax = np.max(obj_pc[:, 1])
        zmax = np.max(obj_pc[:, 2])
        bbox = np.array(
            [
                (xmin + xmax) / 2.0,
                (ymin + ymax) / 2.0,
                (zmin + zmax) / 2.0,
                xmax - xmin,
                ymax - ymin,
                zmax - zmin,
                label_id,
            ]
        )
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        instance_bboxes[obj_id - 1, :] = bbox

    return (
        mesh_vertices.astype(np.float32),
        label_ids.astype(np.int),
        instance_ids.astype(np.int),
        instance_bboxes.astype(np.float32),
        object_id_to_label_id,
    )


@dataclass
class ScanNetConfig:
    BASE_URL = "http://kaldir.vc.in.tum.de/scannet/"
    TOS_URL = BASE_URL + "ScanNet_TOS.pdf"
    FILETYPES = [
        ".aggregation.json",
        ".sens",
        ".txt",
        "_vh_clean.ply",
        "_vh_clean_2.0.010000.segs.json",
        "_vh_clean_2.ply",
        "_vh_clean.segs.json",
        "_vh_clean.aggregation.json",
        "_vh_clean_2.labels.ply",
        "_2d-instance.zip",
        "_2d-instance-filt.zip",
        "_2d-label.zip",
        "_2d-label-filt.zip",
    ]
    SPLITS = ["train", "val", "test"]
    FILETYPES_TEST = [".sens", ".txt", "_vh_clean.ply", "_vh_clean_2.ply"]
    PREPROCESSED_FRAMES_FILE = ["scannet_frames_25k.zip", "5.6GB"]
    TEST_FRAMES_FILE = ["scannet_frames_test.zip", "610MB"]
    LABEL_MAP_FILES = ["scannetv2-labels.combined.tsv", "scannet-labels.combined.tsv"]
    RELEASES = ["v2/scans", "v1/scans"]
    RELEASES_TASKS = ["v2/tasks", "v1/tasks"]
    RELEASES_NAMES = ["v2", "v1"]
    RELEASE = RELEASES[0]
    RELEASE_TASKS = RELEASES_TASKS[0]
    RELEASE_NAME = RELEASES_NAMES[0]
    LABEL_MAP_FILE = LABEL_MAP_FILES[0]
    RELEASE_SIZE = "1.2TB"
    V1_IDX = 1
    NUM_CLASSES = 41
    CLASS_LABELS = (
        "wall",
        "floor",
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refrigerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "otherfurniture",
    )
    URLS_METADATA = [
        "https://raw.githubusercontent.com/facebookresearch/votenet/master/scannet/meta_data/scannetv2-labels.combined.tsv",
        "https://raw.githubusercontent.com/facebookresearch/votenet/master/scannet/meta_data/scannetv2_train.txt",
        "https://raw.githubusercontent.com/facebookresearch/votenet/master/scannet/meta_data/scannetv2_test.txt",
        "https://raw.githubusercontent.com/facebookresearch/votenet/master/scannet/meta_data/scannetv2_val.txt",
    ]
    VALID_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]


class ScanNet(Dataset):
    def __init__(self,
                 data_dir,
                 split,
                 normalize_rgb,
                 donotcare_class_ids = [],
                 max_num_point=None,) -> None:
        self.data_dir = data_dir
        self.split = split
        if split == 'test':
            self.scan_dir = os.path.join(data_dir, 'scans_test')
        else:
            self.scan_dir = os.path.join(data_dir, 'scans')
        self.metadata_path = os.path.join(self.data_dir, "metadata")
        self.label_map_file = os.path.join(self.metadata_path,
                                           "scannetv2-labels.combined.tsv")
        split_file = os.path.join(self.metadata_path,
                                  "scannetv2_{}.txt".format(split))
        self.scan_names = self.read_scan_names(split_file)
        self.normalize_rgb = normalize_rgb
        self.donotcare_class_ids = donotcare_class_ids
        self.max_num_point = max_num_point

    def read_scan_names(self, split_file):
        with open(split_file) as f:
            scan_names = sorted([line.rstrip() for line in f])
            f.close()
        return scan_names

    @staticmethod
    def read_one_test_scan(scannet_dir, scan_name, normalize_rgb):
        mesh_file = os.path.join(scannet_dir, scan_name, scan_name + "_vh_clean_2.ply")
        mesh_vertices = read_mesh_vertices_rgb(mesh_file)

        data = {}
        data["pos"] = torch.from_numpy(mesh_vertices[:, :3])
        data["rgb"] = torch.from_numpy(mesh_vertices[:, 3:])
        if normalize_rgb:
            data["rgb"] /= 255.0
        return Data(**data)

    @staticmethod
    def read_one_scan(
        scannet_dir,
        scan_name,
        label_map_file,
        donotcare_class_ids,
        max_num_point,
        normalize_rgb,
    ):
        mesh_file = os.path.join(scannet_dir, scan_name, scan_name + "_vh_clean_2.ply")
        agg_file = os.path.join(scannet_dir, scan_name, scan_name + ".aggregation.json")
        seg_file = os.path.join(
            scannet_dir, scan_name, scan_name + "_vh_clean_2.0.010000.segs.json"
        )
        meta_file = os.path.join(
            scannet_dir, scan_name, scan_name + ".txt"
        )  # includes axisAlignment info for the train set scans.
        mesh_vertices, semantic_labels, instance_labels, _, _ = export(
        mesh_vertices, semantic_labels, instance_labels, _, _ = export(
            mesh_file, agg_file, seg_file, meta_file, label_map_file, None
        )

        # Discard unwanted classes
        mask = np.logical_not(np.in1d(semantic_labels, donotcare_class_ids))
        mesh_vertices = mesh_vertices[mask, :]
        semantic_labels = semantic_labels[mask]
        instance_labels = instance_labels[mask]

        # Subsample
        N = mesh_vertices.shape[0]
        if max_num_point:
            if N > max_num_point:
                choices = np.random.choice(N, max_num_point, replace=False)
                mesh_vertices = mesh_vertices[choices, :]
                semantic_labels = semantic_labels[choices]
                instance_labels = instance_labels[choices]

        # Build data container
        data = {}
        data["pos"] = torch.from_numpy(mesh_vertices[:, :3])
        data["rgb"] = torch.from_numpy(mesh_vertices[:, 3:])
        if normalize_rgb:
            data["rgb"] /= 255.0
        data["y"] = torch.from_numpy(semantic_labels)
        data["x"] = None
        data["instance_labels"] = torch.from_numpy(instance_labels)

        return Data(**data)

    def __getitem__(self, index) -> T_co:
        if self.split == 'test':
            data = ScanNet.read_one_test_scan(self.scan_dir,
                                              self.scan_names[index],
                                              normalize_rgb=self.normalize_rgb)
        else:
            data = ScanNet.read_one_scan(self.scan_dir,
                                         self.scan_names[index],
                                         label_map_file=self.label_map_file,
                                         donotcare_class_ids=self.donotcare_class_ids,
                                         max_num_point=self.max_num_point,
                                         normalize_rgb=self.normalize_rgb
                                         )
        return data

class ScanNetDataModule(Base3dDataModule):
    def __init__(
        self,
        config: DataModuleConfig = ...,
        transforms: DataModuleTransforms = ...,
        dataset_config: ScanNetConfig = ScanNetConfig(),
    ):
        super().__init__(config, transforms)

        self.data_config = dataset_config
        self.data_dir = config.data_dir
        self.normalize_rgb = config.normalize_rgb
        self.donotcare_class_ids = config.donotcare_class_ids
        self.max_num_point = config.max_num_point
        self.data_dir = config.data_dir
        self.normalize_rgb = config.normalize_rgb
        self.donotcare_class_ids = config.donotcare_class_ids
        self.max_num_point = config.max_num_point

        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        """Download ScanNet data."""
        release_file = self.data_config.BASE_URL + self.data_config.RELEASE + ".txt"
        release_scans = get_release_scans(release_file)
        # release_scans = ["scene0191_00","scene0191_01", "scene0568_00", "scene0568_01"]
        file_types = self.data_config.FILETYPES
        release_test_file = self.data_config.BASE_URL + self.data_config.RELEASE + "_test.txt"
        release_test_scans = get_release_scans(release_test_file)
        file_types_test = self.data_config.FILETYPES_TEST
        out_dir_scans = os.path.join(self.data_dir, "scans")
        out_dir_test_scans = os.path.join(self.data_dir, "scans_test")

        if self.types:  # download file type
            file_types = self.types
            for file_type in file_types:
                if file_type not in self.data_config.FILETYPES:
                    log.error("ERROR: Invalid file type: " + file_type)
                    return
            file_types_test = []
            for file_type in file_types:
                if file_type in self.data_config.FILETYPES_TEST:
                    file_types_test.append(file_type)
        download_label_map(self.data_dir)
        print(
            "WARNING: You are downloading all ScanNet "
            + self.data_config.RELEASE_NAME
            + " scans of type "
            + file_types[0]
        )
        print(
            "Note that existing scan directories will be skipped. Delete partially downloaded directories to re-download."
        )
        print("***")
        print("Press any key to continue, or CTRL-C to exit.")
        input("")
        if self.version == "v2" and ".sens" in file_types:
            print(
                "Note: ScanNet v2 uses the same .sens files as ScanNet v1: Press 'n' to exclude downloading .sens files for each scan"
            )
            key = input("")
            if key.strip().lower() == "n":
                file_types.remove(".sens")
        download_release(release_scans, out_dir_scans, file_types, use_v1_sens=True)
        if self.version == "v2":
            download_label_map(self.data_dir)
            download_release(
                release_test_scans, out_dir_test_scans, file_types_test,
                use_v1_sens=True)

    def setup(self, stage: Optional[str] = None):
        self.data_train = ScanNet(self.data_dir,
                                  split='train',
                                  normalize_rgb=self.normalize_rgb,
                                  donotcare_class_ids=self.donotcare_class_ids,
                                  max_num_point=self.max_num_point)
        self.data_val = ScanNet(self.data_dir,
                                split='val',
                                normalize_rgb=self.normalize_rgb,
                                donotcare_class_ids=self.donotcare_class_ids,
                                max_num_point=self.max_num_point)
        self.data_test = ScanNet(self.data_dir,
                                split='test',
                                normalize_rgb=self.normalize_rgb,
                                donotcare_class_ids=self.donotcare_class_ids,
                                max_num_point=self.max_num_point)
