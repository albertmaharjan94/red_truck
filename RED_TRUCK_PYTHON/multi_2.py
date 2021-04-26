from multiprocessing import Process
import time
import threading

processes = []
threads = []
_start = time.perf_counter()
def normal_loop(i):
    for i in range(100):
        print(f"Process {i}, running")
        print(f"Sleep {i} complete")
    
    time.sleep(1)

def thread_test():
    for i in range(9):
        t = threading.Thread(target=normal_loop, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()



if __name__ == '__main__':
    _start = time.perf_counter()
    def multi_processing_test():
        for i in range(2):
            p = Process(target=normal_loop, args=(i,))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()
    multi_processing_test()
    finish = time.perf_counter()
    print(f'Finished in {round(finish-_start,2 )} second(s)')