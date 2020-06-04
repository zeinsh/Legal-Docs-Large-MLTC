from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
from EuroVocAnalyzeTool import Graph, EuroVocAnalyzeTool
import pickle

TMP_PATH = './tmp/EuroVoc/'
eurovocSaveDir = './data/EuroVoc/'

DOMAIN_FILENAME = 'dom_en.xml'
THESAURUS_FILENAME = 'thes_en.xml'
DESCRIPTEUR_FILENAME = 'desc_en.xml'
DESC_THES_FILENAME = 'desc_thes.xml'
USEDFOR_FILENAME = 'uf_en.xml'
SCOPE_NOTE_FILENAME = 'sn_en.xml'
RELATION_UI_FILENAME = 'relation_ui.xml'
RELATION_BT_FILENAME = 'relation_bt.xml'
RELATION_RT_FILENAME = 'relation_rt.xml'


def parseXMLDOMAINS(tmppath, domain_filename='dom_en.xml'):
    COLNAMES = ['Domain_id', 'Label']
    data = pd.DataFrame(columns=COLNAMES)
    id2label = {}

    with open(tmppath + '/' + domain_filename) as fin:
        xmlFile = fin.read()
    parsedXml = BeautifulSoup(xmlFile, features="html.parser")

    records = parsedXml.find_all('record')
    for record in records:
        thes_id = record.find('domaine_id').text
        label = record.find('libelle').text
        data.loc[len(data)] = [thes_id, label]
        id2label[thes_id] = label
    return data, id2label


def parseXMLTHESAURUS(path, thesaurus_filename):
    COLNAMES = ['Thesaurus_id', 'Descriptors', 'Domain_id', 'Domain_Label']
    data = pd.DataFrame(columns=COLNAMES)
    id2label = {}

    with open(path + '/' + thesaurus_filename) as fin:
        xmlFile = fin.read()
    parsedXml = BeautifulSoup(xmlFile, features="html.parser")

    records = parsedXml.find_all('record')
    for record in records:
        thes_id = record.find('thesaurus_id').text
        label = record.find('libelle').text

        domain_id = thes_id[:2]
        domain_label = domain_id2label[domain_id]

        data.loc[len(data)] = [thes_id, label, domain_id, domain_label]
        id2label[thes_id] = label

    return data, id2label


def parseXMLDESCRIPTORS(path, descripteur_filename):
    COLNAMES = ['Descripteur_id', 'Label', 'Def']
    data = pd.DataFrame(columns=COLNAMES)
    id2label = {}

    with open(path + '/' + descripteur_filename) as fin:
        xmlFile = fin.read()
    parsedXml = BeautifulSoup(xmlFile, features="html.parser")

    records = parsedXml.find_all('record')
    for record in records:
        desc_id = record.find('descripteur_id').text
        desc_label = record.find('libelle').text
        desc_def = '' if record.find('def') is None else record.find('def').text

        data.loc[len(data)] = [desc_id, desc_label, desc_def]
        id2label[desc_id] = desc_label
    return data, id2label


def parseXMLDESC_THES(path, desc_thes_filename):
    COLNAMES = ['Thesaurus_id', 'Thesaurus_Label', 'Descripteur_id', 'Descripteur_Label', 'Domain_id', 'Domain_Label',
                'TopTerm', 'is_country']
    data = pd.DataFrame(columns=COLNAMES)
    desc2thes = {}

    with open(path + '/' + DESC_THES_FILENAME) as fin:
        xmlFile = fin.read()
    parsedXml = BeautifulSoup(xmlFile, features="html.parser")

    records = parsedXml.find_all('record')
    for record in records:
        desc_id = record.find('descripteur_id').text
        desc_label = desc_id2label[desc_id]
        thes_id = record.find('thesaurus_id').text
        thes_label = thes_id2label[thes_id]
        domain_id = thes_id[:2]
        domain_label = domain_id2label[domain_id]
        topterm = record.find('topterm').text
        is_country = record.find('descripteur_id').get('country')
        data.loc[len(data)] = [thes_id, thes_label, desc_id, desc_label, domain_id, domain_label, topterm, is_country]

        descnode = desc2thes.get(desc_id, [])
        descnode.append(thes_id)
        desc2thes[desc_id] = descnode
    return data, desc2thes


def parseXMLUSEDFOR(path, usedfor_filename):
    COLNAMES = ['Descripteur_id', 'Descripteur_Label', 'UsedForElements']
    data = pd.DataFrame(columns=COLNAMES)
    desc_usedfor = {}

    with open(path + '/' + usedfor_filename) as fin:
        xmlFile = fin.read()
    parsedXml = BeautifulSoup(xmlFile, features="html.parser")

    records = parsedXml.find_all('record')
    for record in records:
        desc_id = record.find('descripteur_id').text
        desc_label = desc_id2label[desc_id]
        usedfor_elements = ';'.join([el.text for el in record.find('uf').find_all('uf_el')])
        data.loc[len(data)] = [desc_id, desc_label, usedfor_elements]
        desc_usedfor[desc_id] = usedfor_elements
    return data, desc_usedfor


def parseXMLSCOPENOTE(path, scope_note_filename):
    COLNAMES = ['Descripteur_id', 'ScopeNote', 'HistoryNote']
    data = pd.DataFrame(columns=COLNAMES)

    with open(path + '/' + scope_note_filename) as fin:
        xmlFile = fin.read()
    parsedXml = BeautifulSoup(xmlFile, features="html.parser")

    records = parsedXml.find_all('record')
    for record in records:
        desc_id = record.find('descripteur_id').text
        scopenote = '' if record.find('sn') is None else record.find('sn').text
        history_note = '' if record.find('hn') is None else record.find('hn').text
        data.loc[len(data)] = [desc_id, scopenote, history_note]

    return data


def parseXMLRELATION_AI(path, relation_ui_filename):
    COLNAMES = ['Source_id', 'Source_Label', 'Target_id', 'Target_Label']
    data = pd.DataFrame(columns=COLNAMES)
    adjacency_list = {}
    with open(path + '/' + relation_ui_filename) as fin:
        xmlFile = fin.read()
    parsedXml = BeautifulSoup(xmlFile, features="html.parser")

    records = parsedXml.find_all('record')
    for record in records:
        source_id = record.find('source_id').text
        target_id = record.find('cible_id').text
        source_label = desc_id2label[source_id]
        target_label = desc_id2label[target_id]
        data.loc[len(data)] = [source_id, source_label, target_id, target_label]

        source_node = adjacency_list.get(source_id, [])
        source_node.append(target_id)

        adjacency_list[source_id] = source_node
        # adjacency_list[target_id]=target_node
    return data, Graph(adjacency_list, desc_id2label)


def parseXMLRELATION_BT(path, relation_bt_filename):
    COLNAMES = ['Source_id', 'Source_Label', 'Target_id', 'Target_Label']
    data = pd.DataFrame(columns=COLNAMES)
    adjacency_list = {}

    with open(path + '/' + relation_bt_filename) as fin:
        xmlFile = fin.read()
    parsedXml = BeautifulSoup(xmlFile, features="html.parser")

    records = parsedXml.find_all('record')
    for record in records:
        source_id = record.find('source_id').text
        target_id = record.find('cible_id').text
        source_label = desc_id2label[source_id]
        target_label = desc_id2label[target_id]
        data.loc[len(data)] = [source_id, source_label, target_id, target_label]

        source_node = adjacency_list.get(source_id, [])
        source_node.append(target_id)
        adjacency_list[source_id] = source_node
    return data, Graph(adjacency_list, desc_id2label)


def parseXMLRELATION_RT(path, relation_rt_filename):
    COLNAMES = ['Source_id', 'Source_Label', 'Target_id', 'Target_Label']
    data = pd.DataFrame(columns=COLNAMES)
    adjacency_list = {}

    with open(path + '/' + relation_rt_filename) as fin:
        xmlFile = fin.read()
    parsedXml = BeautifulSoup(xmlFile, features="html.parser")

    records = parsedXml.find_all('record')
    for record in records:
        source_id = record.find('descripteur1_id').text
        target_id = record.find('descripteur2_id').text
        source_label = desc_id2label[source_id]
        target_label = desc_id2label[target_id]
        data.loc[len(data)] = [source_id, source_label, target_id, target_label]

        source_node = adjacency_list.get(source_id, [])
        source_node.append(target_id)
        adjacency_list[source_id] = source_node
    return data, Graph(adjacency_list, desc_id2label)


if __name__ == "__main__":
    print("Prepare Graph!")
    print("Load Domains ...")
    Path(eurovocSaveDir).mkdir(parents=True, exist_ok=True)
    xml_domains, domain_id2label = parseXMLDOMAINS(TMP_PATH)
    xml_domains.to_csv(eurovocSaveDir + 'dom_en.csv')

    # thes_en.xml
    print("Load M-Thesaurus ...")
    xml_thesaurus, thes_id2label = parseXMLTHESAURUS(TMP_PATH, THESAURUS_FILENAME)
    xml_thesaurus.to_csv(eurovocSaveDir + 'thes_en.csv')

    print("Load Descriptors ...")
    xml_descriptors, desc_id2label = parseXMLDESCRIPTORS(TMP_PATH, DESCRIPTEUR_FILENAME)
    xml_descriptors.to_csv(eurovocSaveDir + 'desc_en.csv')

    print("Load Descriptor - MThesaurus relation ...")
    xml_desc_thes, desc2thes = parseXMLDESC_THES(TMP_PATH, DESC_THES_FILENAME)
    xml_desc_thes.to_csv(eurovocSaveDir + 'desc_thes.csv')

    print("Preparing Topterms ...")
    topterms = set(xml_desc_thes[xml_desc_thes['TopTerm'] == 'O']['Descripteur_id'].unique())

    # uf_en.xml
    print("Load Used-for relation ...")
    xml_descriptors_usedfor, desc_usedfor = parseXMLUSEDFOR(TMP_PATH, USEDFOR_FILENAME)
    xml_descriptors_usedfor.to_csv(eurovocSaveDir + 'desc_uf_en.csv')

    # sn_en.xml
    print("Load scope-note ...")
    xml_descriptors_scopenote = parseXMLSCOPENOTE(TMP_PATH, SCOPE_NOTE_FILENAME)
    xml_descriptors_scopenote.to_csv(eurovocSaveDir + 'desc_sn_en.csv')

    # relation_ui.xml
    print("Load used-instead relation ...")
    relation_ui, graph_ui = parseXMLRELATION_AI(TMP_PATH, RELATION_UI_FILENAME)
    relation_ui.to_csv(eurovocSaveDir + 'relation_used_instead.csv')

    # relation_bt.xml
    print("Load broader relation ...")
    relation_bt, graph_bt = parseXMLRELATION_BT(TMP_PATH, RELATION_BT_FILENAME)
    relation_bt.to_csv(eurovocSaveDir + 'relation_broader.csv')

    # relation_rt.xml
    print("Load related relation ...")
    relation_rt, graph_rt = parseXMLRELATION_RT(TMP_PATH, RELATION_RT_FILENAME)
    relation_rt.to_csv(eurovocSaveDir + 'relation_related.csv')

    analyzeTool = EuroVocAnalyzeTool(domain_id2label, thes_id2label, desc_id2label,
                                     desc2thes, topterms, desc_usedfor,
                                     graph_ui, graph_bt, graph_rt)

    print("Dump EuroVoc analysis tool ...")
    with open('data/EuroVocAnalysisTool.pickle', 'wb') as handle:
        pickle.dump(analyzeTool, handle, protocol=pickle.HIGHEST_PROTOCOL)
