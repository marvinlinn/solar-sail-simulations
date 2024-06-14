import numpy as np
import multiprocessing
import time

def worker(number, queue):
    print(f'Worker {number} started')
    time.sleep(2)
    res = (np.random.random((3,3)), np.ones((3,4)))
    queue.put(res)
    print(f'Worker {number} finished')
    return res

if __name__ == '__main__':
    queue = multiprocessing.Queue(maxsize=10)
    processes = []

    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i, queue))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
     
    result = queue.get(timeout=5)
    print(result)
            
    print('All processes completed')
