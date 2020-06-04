import copy # dict = copy.deepcopy(dict)
import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, adj_list_dict, id2label):
        self.graph=adj_list_dict
        self.inv_graph=self.__inverted_graph__(self.graph)
        self.bi_graph=self.__bidirected_graph__(self.graph)
        self.id2label=id2label
    def __inverted_graph__(self, graph_rt):
        inverted_graph={}
        for source_id in graph_rt.keys():
            source_node=graph_rt[source_id]
            for target_id in source_node:
                target_node=inverted_graph.get(target_id,[])
                target_node.append(source_id)

                inverted_graph[target_id]=target_node
        return inverted_graph
    def __bidirected_graph__(self, graph_rt):
        bidirected_graph={}
        for source_id in graph_rt.keys():
            source_node=graph_rt[source_id]
            for target_id in source_node:
                # add inverted
                target_node=bidirected_graph.get(target_id,[])
                target_node.append(source_id)
                bidirected_graph[target_id]=target_node
                
                # add direct  
                target_node=bidirected_graph.get(source_id,[])
                target_node.append(target_id)
                bidirected_graph[source_id]=target_node
                
        return bidirected_graph
    def get_nodes(self):
        return list(self.bi_graph.keys())    
    def get_edges_list(self):
        edges=[]
        for source_id in self.graph.keys():
            source_node=self.graph[source_id]
            for target_id in source_node:
                edges.append((source_id, target_id))
        return edges
    def dfs(self, start, inverted=False, bidirected=False):
        graph_rt=self.graph if not inverted else self.inv_graph
        if bidirected:
            graph_rt=self.bi_graph
        elif inverted:
            graph_rt=self.inv_graph
        else:
            graph_rt=self.graph
        
        related=[]
        related_graph={}

        stack=[start]
        found=[start]
        processed=[]
        while len(stack) > 0:
            
            current=stack.pop()
            related.append(current)
            processed.append(start)

            neighbours=graph_rt.get(current,[])
            related_graph[current]=copy.deepcopy(neighbours)
            for neighbour in neighbours:
                if neighbour not in found:
                    stack.append(neighbour)
                    found.append(neighbour)

        return Graph(related_graph, self.id2label)
    def print_edges(self):
        edges=self.get_edges_list()
        print("Relations in related_graph\n",'='*15)
        for p, c in edges:
            print("{}\t{}\t{}\t{}".format(p,c,self.id2label[p],self.id2label[c]))
    def draw_graph(self, start):
        edges=self.get_edges_list()
        G=nx.Graph()
        G.add_node(start)
        G.add_edges_from(edges)
        nx.draw(G, with_labels=True, node_size=2000)
        
## Analyze Tool
class EuroVocAnalyzeTool:
    def __init__(self, domain_id2label, thesaurus_id2label, descripteur_id2label,
                 desc2thes, topterms, desc_usedfor,
                 graph_ui, graph_bt, graph_rt):
        
        self.domain_id2label=domain_id2label
        self.thesaurus_id2label=thesaurus_id2label
        self.descripteur_id2label=descripteur_id2label
        self.topterms = topterms
        self.desc2thes=desc2thes
        self.desc_usedfor=desc_usedfor
        
        self.graph_ui=graph_ui
        self.graph_bt=graph_bt
        self.graph_rt=graph_rt
        
    def getDomainLabelById(self, dom_id):
        return self.domain_id2label.get(dom_id,None)
    def getThesaurusLabelById(self, thes_id):
        return self.thesaurus_id2label.get(thes_id,None)
    def getDescLabelById(self, desc_id):
        return self.descripteur_id2label.get(desc_id,None)
    def getThesaurusByDescId(self, desc_id):
        return self.desc2thes.get(desc_id, None)
    def getDomainsByDescId(self, desc_id):
        thes_ids=self.getThesaurusByDescId(desc_id)
        return None if  thes_ids is None else [thes_id[:2] for thes_id in thes_ids]
    def getParents(self, desc_id):
        return self.graph_bt.dfs(desc_id, inverted=False).get_nodes()
    def getTopTermsByDescid(self, desc_id):
        parent_nodes=self.getParents(desc_id)+[desc_id]
        return [node for node in parent_nodes if node in self.topterms]
    def getDescripteurUsedFor(self, desc_id):
        return self.desc_usedfor.get(desc_id,'')
    def areRelated(self, desc_id1, desc_id2):
        related1=self.graph_rt.dfs(desc_id1, inverted=False).get_nodes()
        related2=self.graph_rt.dfs(desc_id2, inverted=False).get_nodes()
        return desc_id1 in related2 or desc_id2 in related1
    def areParentChild(self, desc_id1, desc_id2):
        related1=self.graph_bt.dfs(desc_id1, inverted=False).get_nodes()
        related2=self.graph_bt.dfs(desc_id2, inverted=False).get_nodes()
        return desc_id1 in related2 or desc_id2 in related1
    def sameDomain(self, desc_id1, desc_id2):
        return self.getDomainsByDescId(desc_id1)==self.getDomainsByDescId(desc_id2)
    def sameThesaurus(self, desc_id1, desc_id2):
        return self.getThesaurusByDescId(desc_id1)==self.getThesaurusByDescId(desc_id2)
