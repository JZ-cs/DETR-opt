import time
import torch
import torch.nn as nn
def avgt(time_lis:list, tunit='ms'):
    if(time_lis is None or len(time_lis)==0): return None, tunit
    
    res = sum(time_lis)/len(time_lis)
    if(tunit == 'ms'):return res,tunit
    elif(tunit == 's'): return res/1e3,tunit
    elif(tunit == 'us'): return res * 1e3,tunit
    else:
        raise RuntimeError(f'Unrecognized time unit {tunit}')

class TrackTime:
    def __init__(self):
        self.is_forward = True
        self.iter_cnt = 0
        self.time_list_dic = []
        self.summary_dic = None

        self.start_event_dic = {}
        self.end_event_dic = {}

        self.cpu_time_start_dic = {}
    
    def init__(self):
        self.is_forward = True
        self.iter_cnt = 0
        self.time_list_dic = []
        self.summary_dic = None

        self.start_event_dic = {}
        self.end_event_dic = {}

    def end_iter(self):
        self.is_forward = True
        self.iter_cnt += 1

    def summary_all(self):
        self.summary_dic = {}
        for it, time_dic in enumerate(self.time_list_dic):
            for kstr, time_lis in time_dic.items():
                if(kstr not in self.summary_dic):
                    self.summary_dic[kstr] = []
                self.summary_dic[kstr].append(sum(time_lis))

    def add_time_interval(self, key_str, time_cost):
        if(self.is_forward == False): return
        if(self.iter_cnt >= len(self.time_list_dic)):
            self.time_list_dic.extend([{} for _ in range(16)])
        
        idx = self.iter_cnt
        if(key_str not in self.time_list_dic[idx].keys()):
            self.time_list_dic[idx][key_str] = []
        self.time_list_dic[idx][key_str].append(time_cost)

    def print_iter_status(self, it = -1):
        if(it == -1):
            print('iter is -1, Do Nothing!')
            return
        for kstr, time_lis in self.time_list_dic[it].items():
            print(f'{kstr}(i={it}): time is {sum(time_lis)}')

    def print_end_status(self, st_idx = 30):
        print('-'*50 + ' endgame ' + '-'*50)
        print(ms_helper.running_stat())
        for kstr, time_lis in self.summary_dic.items():
            print(f'{kstr}({len(time_lis[st_idx:])} iters): avg time is {avgt(time_lis[st_idx:])}')
        print('-'*50 + ' endgame ' + '-'*50)
    
    def cuda_record_start(self, kstr):
        if(kstr not in self.start_event_dic.keys()):
            self.start_event_dic[kstr] = torch.cuda.Event(enable_timing=True)
            self.end_event_dic[kstr] = torch.cuda.Event(enable_timing=True)
        self.start_event_dic[kstr].record()
    
    def cuda_record_end(self, kstr):
        self.end_event_dic[kstr].record()
        torch.cuda.synchronize()
        _time = self.start_event_dic[kstr].elapsed_time(self.end_event_dic[kstr])
        return _time
    
    def cpu_record_start(self, kstr):
        if(kstr not in self.cpu_time_start_dic.keys()):
            self.cpu_time_start_dic[kstr] = 0.
        torch.cuda.synchronize()
        self.cpu_time_start_dic[kstr] = time.perf_counter_ns()
    
    def cpu_record_end(self, kstr):
        torch.cuda.synchronize()
        end_t = time.perf_counter_ns()
        return (end_t - self.cpu_time_start_dic[kstr]) / 1e6
        

    def search_module(self, mod:nn.Module, target):
        if(mod.__class__ == target):
            assert(isinstance(mod, target))
            return 1
        mod_num = 0
        for name, sub_m in mod.named_children():
            mod_num += self.search_module(sub_m, target)
        return mod_num

    def serach_conv(self, mod:nn.Module):
        if(isinstance(mod, nn.Conv2d)):
            return 1
        convs = 0
        for name, m in mod.named_children():
            convs += self.serach_conv(m)
        return convs
    
    def search_one_layer(self, layer:nn.Module):
        if(self.is_forward==False or layer in self.searched):
            return
        self.splt_conv_list.append(self.serach_conv(layer))
        self.searched.add(layer)


class Stream_wrapper:
    """ A Stream_wrapper contains a few cuda streams, which are defined with 
        torch.cuda.Stream(device),
        and their corresponding cuda device"""
    def __init__(self, device, n_streams=2) -> None:
        self.device = device
        self.cuda_streams = [torch.cuda.Stream(device=device) for _ in range(n_streams)]
        self.execute_cnt = [0 for _ in range(n_streams)]
    def __getitem__(self, idx):
        return self.cuda_streams[idx]
    def __setitem__(self, idx, val):
        assert(isinstance(val, torch.cuda.Stream))
        self.cuda_streams[idx] = val
    def __repr__(self):
        streams_list = ['({})'.format(cuda_stream) for cuda_stream in self.cuda_streams]
        return ('<Stream_Wrapper on device= {} cuda_streams= {}>'
                .format(self.device, streams_list))     

class Multi_stream_helper:
    """ Multi_stream_helper is used as a global variable to provide information
        when using multi-stream"""
    def __init__(self) -> None:
        self.use_multi_stream = False
        self.fuse_detr = False
        self.prep_ip_opt = False
        self.input_opt = False
        self.gt_opt = False
        self.nvtx_profile = False
        self.stream_0 = torch.cuda.Stream(device=0)
        self.stream_1 = torch.cuda.Stream(device=0)
        self.stream_2 = torch.cuda.Stream(device=0)
        # self.stream_3 = torch.cuda.Stream(device=0)
        # self.stream_4 = torch.cuda.Stream(device=0)
        # self.hstreams = [torch.cuda.Stream(device=0, priority=-1) for _ in range(5)]
        # self.lstreams = [torch.cuda.Stream(device=0) for _ in range(5)]

        self.forward_modules = None
        self.sw = None
        self.sf_count = {}
    
    def init__(self):
        self.sw = None
        self.sf_count = {}

    def add_sf_count(self, kstr, val=1):
        if(kstr not in self.sf_count.keys()):
            self.sf_count[kstr] = 0
        self.sf_count[kstr] += val
    
    def running_stat(self):
        print_str = ''
        if(self.use_multi_stream==True):
            print_str += 'using Multi-Stream'
        else:
            print_str += 'using Base'
        
        print_str += '\t input_opt:{}\t gtarget_opt:{}\t prep_input:{}\t fuse_detr:{}\t nvtx:{} '.format(
            self.input_opt, self.gt_opt, self.prep_ip_opt, self.fuse_detr, self.nvtx_profile
        )
        return print_str
        

ms_helper = Multi_stream_helper()
Stream_wrapper_dict = {}
tracktime = TrackTime()  


