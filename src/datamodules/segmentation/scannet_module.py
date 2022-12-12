import logging
import os
import tempfile
import urllib
from dataclasses import dataclass
from urllib.request import urlopen

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
    # print("Downloaded scan " + scan_id)


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


class ScanNetDataModule(Base3dDataModule):
    def __init__(
        self,
        config: DataModuleConfig = ...,
        transforms: DataModuleTransforms = ...,
        dataset_config: ScanNetConfig = ScanNetConfig(),
    ):
        super().__init__(config, transforms)

        self.data_config = dataset_config

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
        out_dir_scans = os.path.join(self.raw_dir, "scans")
        out_dir_test_scans = os.path.join(self.raw_dir, "scans_test")

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
        download_label_map(self.raw_dir)
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
            download_label_map(self.raw_dir)
            download_release(
                release_test_scans, out_dir_test_scans, file_types_test, use_v1_sens=True
            )
