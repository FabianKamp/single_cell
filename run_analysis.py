from single_cell import single_cell

if __name__ == "__main__":
    sc = single_cell()
    sc.convert_data()
    sc.seg_data()