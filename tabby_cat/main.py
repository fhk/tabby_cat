"""
Main entry point for tabby cat
"""

from tabby_cat.data_loader import DataLoader
from tabby_cat.processor import Processor


def main():
    where = "Alaska"
    dl = DataLoader()
    dl.download_data_geofabrik(where)
    dl.read_street_data(where)
    dl.download_data_openaddress(where)
    dl.read_address_data(where)
    #dl.address_df.to_file("address.shp")

    pr = Processor()
    pr.snap_points_to_line(dl.streets_df, dl.address_df)

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
