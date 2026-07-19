import os
from ipaddress import ip_address

# 【关键修改 1】指向当前编译的 P4 程序名 delay_measure_main
p4 = bfrt.delay_measure_main.pipe

# 清理残留表项的基础函数
def clear_all(verbose=True, batching=True):
    global p4
    global bfrt
    
    def _clear(table, verbose=False, batching=False):
        if verbose:
            print("Clearing table {:<40} ... ".
                  format(table['full_name']), end='', flush=True)
        try:    
            entries = table['node'].get(regex=True, print_ents=False)
            try:
                if batching:
                    bfrt.batch_begin()
                for entry in entries:
                    entry.remove()
            except Exception as e:
                print("Problem clearing table {}: {}".format(
                    table['name'], e.sts))
            finally:
                if batching:
                    bfrt.batch_end()
        except Exception as e:
            if e.sts == 6:
                if verbose:
                    print('(Empty) ', end='')
        finally:
            if verbose:
                print('Done')

        try:
            table['node'].reset_default()
        except:
            pass

    for table in p4.info(return_info=True, print_info=False):
        if table['type'] in ['MATCH_DIRECT', 'MATCH_INDIRECT_SELECTOR']:
            _clear(table, verbose=verbose, batching=batching)
    for table in p4.info(return_info=True, print_info=False):
        if table['type'] in ['SELECTOR']:
            _clear(table, verbose=verbose, batching=batching)
    for table in p4.info(return_info=True, print_info=False):
        if table['type'] in ['ACTION_PROFILE']:
            _clear(table, verbose=verbose, batching=batching)

# 1. 首先清空机器上残留的旧表项，防止冲突
clear_all()

# 2. 获取 P4 代码中定义的 port_match 表节点
port_match = p4.Ingress.port_match
port_match.clear()

# =================================================================
# 【关键修改 2】配置 U-Turn (原路返回) 转发规则
# 基于真实的 DP 端口号: 176 (7/0), 178 (7/2), 179 (7/3)
# 这样 sender.go 发出的包才能被反射回原服务器进行抓包解析
# =================================================================

print("Installing U-Turn routing rules...")
port_match.add_with_send(ingress_port=176, port=176)
port_match.add_with_send(ingress_port=178, port=178)
port_match.add_with_send(ingress_port=179, port=179)

# 设置默认兜底动作：所有未匹配规则的包，默认发往 176 端口
# （您可以根据发包机实际连接的端口修改此默认值）
port_match.set_default_with_send(port=176)

# 3. 提交 BFRT 操作到数据平面
bfrt.complete_operations()

# 4. 打印最终下发的流表状态以供确认
print("\n******************* PROGAMMING RESULTS *****************")
print("Table port_match:")
port_match.dump(table=True)