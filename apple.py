# Apple Machine Learning Research Engineer


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
