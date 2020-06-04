from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import mode
import numpy as np

class HTMLTable:
    def __init__(self, title, columns):
        self.columns=columns
        self.data=[]
        self.title=title
    def addRow(self, row):
        # row is list
        assert type(row)==type([])
        assert len(row)<=len(self.columns)
        self.data.append(row)
    def getHtmlTable(self):
        ret='<h3>{}</h3>'.format(self.title)
        ret+='<table>'
        
        for field in self.columns:
            ret+='<td><B>{}</B></td>'.format(field)
            
        for row in self.data:
            ret+='<tr>'
            for field in row:
                ret+='<td>{}</td>'.format(field)
            ret+='</tr>'
        ret+='</table>'
        return ret

def getStatisticsReportHTML(title, arr):
    htmltable=HTMLTable(title,['stat','value'])
    htmltable.addRow(['Number of elements',len(arr)])
    htmltable.addRow(['Max',np.max(arr)])
    htmltable.addRow(['Min',np.min(arr)])
    htmltable.addRow(['Mean',round(np.mean(arr),4)])
    htmltable.addRow(['std. dev.',round(np.std(arr),4)])
    htmltable.addRow(['Median',np.median(arr)])
    htmltable.addRow(['Mode value',mode(arr)[0][0]])
    htmltable.addRow(['Mode count',mode(arr)[1][0]])
    htmltable.addRow(['kurtosis',round(kurtosis(arr),4)])
    htmltable.addRow(['skew',round(skew(arr),4)])
    htmltable.addRow(['sum',round(np.sum(arr),4)])
    htmltable.addRow(['unique',len(set(arr))])
    return htmltable.getHtmlTable()
