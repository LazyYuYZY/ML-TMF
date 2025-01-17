from rw_files import *


if __name__ == '__main__':
    # for T_slice in [5]:
    #     folder_path = "./traindata_set_"+str(T_slice)+"s/traindata_flows_"+str(T_slice)+"s/"
    #     rw_files.change_name(folder_path, old_part="s_", new_part="",del_l=True)
    #     rw_files.change_name(folder_path, old_part="_20231101", new_part=".pcap", del_r=True)
    for T_slice in [5]:
        # folder_path = "./testdata3_set_"+str(T_slice)+"s/testdata_pcap_"+str(T_slice)+"s/"
        folder_path = "./caida/testdata_pcap_" + str(T_slice) + "s/"
        rw_files.change_name(folder_path, old_part="s_", new_part="",del_l=True)
        # rw_files.change_name(folder_path, old_part="_20230103", new_part=".pcap", del_r=True)
        rw_files.change_name(folder_path, old_part="_2019", new_part=".pcap", del_r=True)