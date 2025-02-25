import random
# Apple Machine Learning Research Engineer


# interview-1
# find the isolated island in a 2D matrix
# grph = ["RRRRE", "RRERE", "RREEE", "EEEEE"]
def count(graph):

    # convert the graph to a 2D list
    def convert_graph(graph):
        return [list(row) for row in graph]

    # erase the "R"
    def erase(graph, i, j):
        if i < 0 or i >= len(graph) or j < 0 or j >= len(graph[0]) or graph[i][j] == 'E':
            return
        if graph[i][j] == 'R':
            graph[i][j] = 'E'
        erase(graph, i+1, j)
        erase(graph, i-1, j)
        erase(graph, i, j+1)
        erase(graph, i, j-1)    

    graph = convert_graph(graph)
    res = 0
    for i in range(len(graph)):
        for j in range(len(graph[0])):
            if graph[i][j] == 'R':
                res += 1
                erase(graph, i, j)
    return res


# test case
graph = ["RRRRE", "RRERE", "RREEE", "EEEEE"]
print(count(graph))     # 1


# interview-2:
# find the top k frequent queries in the list of queries
# queries = ["tiktok", "apple", "meta", "tiktok", "apple", "apple"], k = 2
def findFrequentQueries(queries, k):
    # convert into dict
    query_dict = {}
    for query in queries:
        if query not in query_dict:
            query_dict[query] = 0
        query_dict[query] += 1
    print(f'query_dict={query_dict}')

    # quick select find the k
    nums = []
    for freq in query_dict.values():
        nums.append(freq)
    print(f'nums={nums}')
    kth_freq = quick_select(nums, len(nums)-k)

    # find the top k queries
    res =[]
    for query, freq in query_dict.items():
        if freq >= kth_freq:
            res.append(query)
    return res


def quick_select(nums, k):
    small, equal, big = [], [], []
    pivot = random.choice(nums)
    for num in nums:
        if num < pivot:
            small.append(num)
        elif num == pivot:
            equal.append(num)
        else:
            big.append(num)
    if k <= len(small):
        return quick_select(small, k)
    elif k <= len(small) + len(equal):
        return quick_select(equal, k-len(small))
    return pivot

