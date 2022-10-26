from timeit import default_timer as timer



def measure_run_time(func):
    def wrapper(*args):
        start = timer()
        func(*args)
        end = timer()
        print(f"{func} took {round(end - start, 3)} seconds")
    return wrapper

@measure_run_time
def test(n):
    for i in range(n):
        continue
    print("done looping...")

test(100_000)


