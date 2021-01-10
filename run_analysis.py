from single_cell import single_cell

if __name__ == "__main__":
    sc = single_cell()
    #sc.setup_folders()
    #sc.convert_data()
    #sc.seg_data()
    #sc.prepare_extr()
    #sc.create_config()
    #sc.plot_segments()
    sc.extr_features()