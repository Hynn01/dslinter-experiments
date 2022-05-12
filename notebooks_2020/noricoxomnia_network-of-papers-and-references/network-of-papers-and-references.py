#!/usr/bin/env python
# coding: utf-8

# # Network of papers and references
# 
# Papers reference other papers, and together they form a network. This network contains various kinds of information. For example, it can be used to find the most influential papers. Or when looking at a specific paper, you can find papers that indirectly reference it or are indirectly referenced by it. When looking at multiple papers, you can see how they are connected. More complex information can also be extracted.
# 
# About two thirds of the papers in the dataset have json files that include the references. This notebook extracts those references and saves the network. It also shows two examples of how the network can be used in other tools.
# 
# The network is a graph with the papers as nodes and the references as directed edges. The papers are identified by their titles, and the references are identified by a pair of titles. The network can be imported in graph tools such as Neo4j or Gephi, which can be used to query various aspects of the network and visualise it.
# 
# The repository with the code for this project can be found on [GitLab](https://gitlab.com/norico.groeneveld/covid-19-research-graph).
# 
# You can also go straight to [the visualisation](https://noricogroeneveld.gitlab.io/covid-19-research-visualisation/).

# In[ ]:


from datetime import datetime
import os
import pandas as pd
import ujson

INPUT_PATH = '../input/CORD-19-research-challenge/'
OUTPUT_PATH = ''

# function to log status, combined with time since start
start_time = datetime.now()
def print_status(message):
    seconds = (datetime.now() - start_time).seconds
    minutes = seconds // 60
    seconds = seconds - (minutes * 60)
    print(f'{minutes:02}:{seconds:02} - {message}')


# # Loading
# In order to add the references to the graph, we need to read the json files. However, not all papers have a json file, and some papers have two. When there are two json files, the one that is parsed from xml is preferred over the one parsed from pdf. We add two columns to the metadata dataframe. One is the best available parse, the other is the path to the best available json file.

# In[ ]:


def load_metadata():
    metadata = pd.read_csv(INPUT_PATH + 'metadata.csv')
    return metadata


def print_parse_counts(metadata):
    group_columns = ['has_pdf_parse', 'has_pmc_xml_parse']
    select_columns = group_columns + ['cord_uid']
    counts = metadata[select_columns].groupby(group_columns).count()
    counts = counts.rename(columns={'cord_uid': 'count'})
    print(counts)
    print()
    print(metadata['best_parse'].value_counts())


def add_best_json_paths(metadata):
    metadata['best_path'] = ''

    metadata['best_parse'] = 'none'
    metadata.loc[metadata['has_pdf_parse'], 'best_parse'] = 'pdf_json'
    metadata.loc[metadata['has_pmc_xml_parse'], 'best_parse'] = 'pmc_json'

    # deal with duplicate pdf hashes
    duplicate_pdf = metadata['sha'].str.contains('; ', regex=False) == True
    first_pdf = metadata.loc[duplicate_pdf, 'sha'].str.split('; ').str[0]
    metadata['first_pdf'] = metadata['sha']
    metadata.loc[duplicate_pdf, 'first_pdf'] = first_pdf
    
    # set folder paths for rows with any parse
    mask = metadata['best_parse'] != 'none'
    metadata.loc[mask, 'best_path'] =        INPUT_PATH +        metadata.loc[mask, 'full_text_file'] + '/' +        metadata.loc[mask, 'full_text_file'] + '/' +        metadata.loc[mask, 'best_parse'] + '/'

    # add pdf filename to paths
    mask = (metadata['best_parse'] == 'pdf_json')
    metadata.loc[mask, 'best_path'] += metadata.loc[mask, 'first_pdf'] + '.json'

    # add pmc filename to paths
    mask = metadata['best_parse'] == 'pmc_json'
    metadata.loc[mask, 'best_path'] += metadata.loc[mask, 'pmcid'] + '.xml.json'

    metadata = metadata.drop(columns=['first_pdf'])
    return metadata


metadata = load_metadata()
metadata = add_best_json_paths(metadata)
print_status('loaded metadata and added json paths')
print_parse_counts(metadata)


# The graph information we can get from purely the metadata.csv file is just the titles. Because the papers with a parse will be added later, we only need the papers without a parse.

# In[ ]:


def titles_to_dataframe(titles, paper_type):
    df = pd.DataFrame(titles, columns=['title', 'publish_time'])
    df['type'] = paper_type
    return df


def nodes_from_metadata(metadata):
    titles = metadata.loc[metadata['best_parse'] == 'none', 'title']
    titles = titles[~pd.isna(titles)]
    titles = titles_to_dataframe(titles, 'no_parse')
    return titles


meta_nodes = nodes_from_metadata(metadata)
print_status(f'extracted {meta_nodes.shape[0]} nodes from metadata')


# Before we actually load the json files, we need some helper functions. Because the title in the json file is not always clean, the title from the metadata file is preferred. When there is no title in either of those, we add an id that is always present, so the references can still be counted.

# In[ ]:


def load_json(path):
    with open(path) as f:
        data = ujson.load(f)
        return data

    
def title_from_paper(metadata_row, json_data):
    paper_title = getattr(metadata_row, 'title')
    if paper_title is None or paper_title == '':
        paper_title = json_data['metadata']['title']
    if paper_title is None or paper_title == '':
        paper_title = getattr(metadata_row, 'cord_uid')
    return paper_title


def title_set_to_dataframe(title_set, paper_type):
    return titles_to_dataframe(list(title_set), paper_type)


# Now we can load the nodes and edges from the json files. In order to filter the papers later, we keep track of the parse.

# In[ ]:


def nodes_and_edges_from_json(metadata):
    titles_pdf_parse = set()
    titles_pmc_parse = set()
    titles_outside = set()
    references = set()

    # handle papers with a json file
    with_json = metadata[metadata['best_path'] != '']
    for row in with_json.itertuples():
        # read paper
        paper_path = getattr(row, 'best_path')
        paper = load_json(paper_path)

        # get and store title
        paper_title = title_from_paper(row, paper)
        paper_publish_time = getattr(row, 'publish_time')
        paper_data = (paper_title, paper_publish_time)
        paper_parse = getattr(row, 'best_parse')
        if paper_parse == 'pdf_json':
            titles_pdf_parse.add(paper_data)
        elif paper_parse == 'pmc_json':
            titles_pmc_parse.add(paper_data)

        # store references (as tuples in set) and possible new titles
        for reference in paper['bib_entries']:
            ref_title = paper['bib_entries'][reference]['title']
            ref_data = (ref_title, '')

            if ref_title != paper_title:
                references.add((paper_title, ref_title))
                titles_outside.add(ref_data)

    # create dataframe with titles of various types
    titles_pdf_parse = title_set_to_dataframe(titles_pdf_parse, 'pdf_parse')
    titles_pmc_parse = title_set_to_dataframe(titles_pmc_parse, 'pmc_parse')
    titles_outside = title_set_to_dataframe(titles_outside, 'outside')
    titles = pd.concat([titles_pdf_parse, titles_pmc_parse, titles_outside])

    references = pd.DataFrame(references, columns=['title', 'referencedTitle'])
    return titles, references


json_nodes, edges = nodes_and_edges_from_json(metadata)
print_status(f'loaded {json_nodes.shape[0]} nodes and {edges.shape[0]} edges from json files')


# # Cleaning
# The titles sometimes contain characters that result in mismatches or may cause problems later (such as \ and quotes), so we want to remove those. Because various kinds of dashes are used, we replace them with the regular dash.

# In[ ]:


REGEX_REMOVE = '[' + ':_()\n' + '\\\\' + '’\'“”\"' + ']'
REGEX_DASH = ' ?[—–-] ?'

def clean_title_series(title):
    title = title        .str.lower()        .str.replace(REGEX_REMOVE, '')        .str.replace(REGEX_DASH, '-')
    return title


def clean_edge_titles(edges):
    edges['title'] = clean_title_series(edges['title'])
    edges['referencedTitle'] = clean_title_series(edges['referencedTitle'])
    edges = edges[edges['title'] != edges['referencedTitle']]
    return edges

meta_nodes['original_title'] = meta_nodes['title']
json_nodes['original_title'] = json_nodes['title']
meta_nodes['title'] = clean_title_series(meta_nodes['title'])
json_nodes['title'] = clean_title_series(json_nodes['title'])
edges = clean_edge_titles(edges)
print_status('cleaned titles')


# Now we need to combine the two dataframes with nodes, after which we can remove duplicates and empty titles. When the title has been added with different parses, we want to keep the parse with the highest quality.

# In[ ]:


def deduplicate_nodes(all_nodes):
    # remove complete duplicates
    all_nodes = all_nodes.drop_duplicates(subset=['title', 'type'])

    # remove duplicates for titles with multiple types
    # keep the row with the higher quality type
    # the highest quality is 'pmc_parse', so it's never removed
    type_order = ['outside', 'no_parse', 'pdf_parse']
    for worse_type in type_order:
        with_multiple_types = all_nodes.duplicated(subset=['title'], keep=False)
        is_worse_type = all_nodes['type'] == worse_type
        remove = with_multiple_types & is_worse_type
        all_nodes = all_nodes[~remove]

    return all_nodes


def index_of_na_or_empty(series):
    return (pd.isna(series)) | (series == '')


all_nodes = pd.concat([meta_nodes, json_nodes])
all_nodes = all_nodes[~index_of_na_or_empty(all_nodes['title'])]
all_nodes = deduplicate_nodes(all_nodes)


# We also need to remove empty titles and duplicates from the edges.

# In[ ]:


def edges_not_na_or_empty(edges, columns):
    for column in columns:
        na_or_empty = index_of_na_or_empty(edges[column])
        edges = edges[~na_or_empty]
    return edges


edges = edges_not_na_or_empty(edges, ['title', 'referencedTitle'])
edges = edges.drop_duplicates(subset=['title', 'referencedTitle'])


# # Filtering
# Large graphs are hard to visualise and are resource-intensive. For some cases, it's better to work with only part of the graph. Here we only use nodes that are referenced at least twice, and remove any edges outside of these nodes. The code can be used with other threshold too, and to filter out papers outside the dataset.

# In[ ]:


def nodes_in_dataset(nodes):
    outside = nodes['type'] == 'outside'
    return nodes[~outside]


def add_reference_counts(nodes, edges):
    counts = edges.groupby('referencedTitle').agg('count')
    counts = counts.reset_index()
    counts = counts.rename(columns={'title': 'times_referenced', 'referencedTitle': 'title'})
    counts['times_referenced'] = counts['times_referenced'].astype('Int32')
    nodes = nodes.merge(counts, on='title', how='left')
    return nodes


def referenced_nodes(nodes, threshold):
    return nodes[nodes['times_referenced'] > threshold]


def filter_edges(filtered_nodes, edges):
    present = pd.DataFrame(filtered_nodes['title'])
    present['present'] = True

    # remove edges if title is not in filtered_nodes
    edges = edges.merge(present, on='title', how='inner')
    edges = edges[edges['present']]
    edges = edges.drop(columns=['present'])

    # remove edges if referencedTitle is not in filtered_nodes
    present = present.rename(columns={'title': 'referencedTitle'})
    edges = edges.merge(present, on='referencedTitle', how='inner')
    edges = edges[edges['present']]
    edges = edges.drop(columns=['present'])

    return edges


def filter_graph(nodes, edges, in_dataset, threshold):
    if in_dataset:
        nodes = nodes_in_dataset(nodes)
    nodes = add_reference_counts(nodes, edges)
    nodes = referenced_nodes(nodes, threshold)

    edges = filter_edges(nodes, edges)
    return nodes, edges

# exclude papers that are only referenced once
subset_nodes, subset_edges = filter_graph(all_nodes, edges, False, 2)

# exclude papers that are referenced less than 5 times, or are outside the original dataset
subsubset_nodes, subsubset_edges = filter_graph(all_nodes, edges, True, 5)


# # Saving

# In[ ]:


def save_to_csv(df, name, header, label):
    name += label
    if df is not None:
        path = f'{OUTPUT_PATH}{name}.csv'
        df.to_csv(path, index=False, header=header)
        print_status(f'saved file {name}.csv with shape {df.shape}')
    else:
        print_status(f'Did not save file {name}.csv, because the df was None')


def save_for_neo4j(nodes, edges, label):
    nodes = nodes.drop(columns=['original_title'])
    save_to_csv(nodes, 'neo_nodes', False, label)
    save_to_csv(edges, 'neo_edges', False, label)


def save_for_gephi(nodes, edges, label):
    nodes = nodes.rename(columns={'title': 'id'})
    edges = edges.rename(columns={'title': 'source', 'referencedTitle': 'target'})
    save_to_csv(nodes, 'gephi_nodes', True, label)
    save_to_csv(edges, 'gephi_edges', True, label)


def save_graph(nodes, edges, label=''):
    save_for_neo4j(nodes, edges, label)
    save_for_gephi(nodes, edges, label)


# complete graph
save_graph(all_nodes, edges)

# filtered data used for large visualisation
save_graph(subset_nodes, subset_edges, '_2')

# filtered data used for normal visualisation
save_graph(subsubset_nodes, subsubset_edges, '_dataset_5')

print_status('Congratulations, the notebook has finished!')


# Now we have csv files that contain the nodes and edges of the research graph. They are ready to be imported in Neo4j or Gephi for analysis or visualisation. Other tools can be used as well, with the same csv files or slightly modified ones.
# 
# [Neo4j](https://neo4j.com/) is a graph database, which is useful for data with many-to-many relationships. It uses the Cypher Query Language and has a Browser to execute queries and view the results. [Gephi](https://gephi.org/) is a program for the visualisation of graphs. It has an interface that's not very complicated, though the number of options can be overwhelming.
# 
# # Neo4j import
# Create a database and put the csv files in the corresponding import folder. Open Neo4j Browser and execute the following three Cypher queries. You might have to change the filenames.
# ```
# CREATE CONSTRAINT ON (p:Paper) ASSERT p.title IS UNIQUE;
# 
# :auto USING PERIODIC COMMIT
# LOAD CSV FROM 'file:///neo_nodes.csv' AS line
# CREATE (:Paper { title: line[0]});
# 
# :auto USING PERIODIC COMMIT
# LOAD CSV FROM 'file:///neo_edges.csv' AS line
# MATCH (paper1:Paper { title: line[0]}),(paper2:Paper { title: line[1]})
# CREATE (paper1)-[:REFERENCES]->(paper2);
# ```
# After that, all the data is loaded and ready to be queried. As an example, you can find the most referenced papers and the references between them.
# ```
# MATCH ()-[r]->(n)
# WITH n, count(r) as relcnt
# WHERE relcnt > 200
# RETURN n, relcnt;
# ```
# 
# # Gephi import and visualisation
# * From the tools menu, make sure the following plugins are installed: Circle Pack, SigmaJS exporter.
# * From the file menu, open the csv file with nodes and follow the steps. Set the Graph Type to Directed.
# * After that, open the csv file with edges. The only change you need to make is to append it to the current workspace instead of creating a new one.
# * In the Statistics panel, run Modularity. There is no need to use weights, but set the Resolution 2 when you use many papers. The Modularity algorithm clusters the papers into highly connected communities.
# * In the Appearance panel, set the node color to Partition using Modularity Class, optionally change the colors for some classes, and apply it.
# * In the same panel, set the node size to Ranking using referenced, with size 5 to 50 (or 100 if using many papers), and apply it
# * In the Layout panel, select Circle Pack Layout, set Hierarchy1 to Modularity Class and run it.
# * In the Preview tab, set the Border With of Nodes to 0. Refresh to see the preview.
# * From the file menu, export the graph as Sigma.js template. Enable *Include search* and *Group edges by direction*, and set the *Group Selector* To *Modularity class*. Make sure that the option *Replace node ids with numbers* is disabled. 
# * To view the result, run *python -m http.server* in the folder with the Sigma.js template. It can also be uploaded, for example to Github Pages or Gitlab Pages. The nodes are smaller than in Gephi, but you can increase the *maxNodeSize* property in the *config.json* file. 
# 
# 
# In the exported visualisation, you can select a paper and view its properties and connections. It is also possible to search for a paper or select a modularity cluster.
# I created two exports, one with the papers in the dataset that are referenced at least 5 times, and another one with all papers (including those outside the dataset) that are referenced at least twice.
# Here are links to view the [first visualisation](https://noricogroeneveld.gitlab.io/covid-19-research-visualisation/) and the [larger second visualisation](https://noricogroeneveld.gitlab.io/covid-19-research-visualisation/full/). The large visualisation has a large data file, so it takes a while to load and is slower.
# 
# Gephi can also be used to calculate properties of the graph, for example with the Pagerank algorithm.
# 
# 
# # Next steps
# * There are more ways to visualise or query the graph and gain information from it. There is research about research graph, which can help with that.
# * The visualisation can be improved, especially regarding the paper attributes.
# * There are various edge cases that can be dealt with in a better way. It is assumed that titles and references are unique, but this is probably not the case. Duplicate titles might belong to papers that are actually different, but now they are combined. There are also some titles that clearly aren't a normal title for a paper.
# * There is more information in the dataset. The most relevant are authors, laboratories, institutions and journals. They can be added to the network as new nodes and edges, or as extra data for the current nodes.
# * It is possible to extract the sentence that a reference is used in, to give more context about the relationship.
# * Other sources might have extra research on COVID-19 to add to the graph.
# 
# 
# # Related solutions
# For COVID-19 research, there are two related solutions. The [references-based atlas notebook](https://www.kaggle.com/mmoeller/a-references-based-atlas-of-covid-19-research) by Moritz Moeller extracts the references from PubMed and visualises them with Gephi. On [covid.curiosity.ai](https://covid.curiosity.ai/) you can use a graph-based search engine with various exploration features.
# 
# There are also various projects related to research graphs in general, including: [Microsoft Academic Graph](https://www.microsoft.com/en-us/research/project/microsoft-academic-graph/), [OpenAIRE Research Graph](https://www.openaire.eu/blogs/the-openaire-research-graph), [ResearchGraph](https://researchgraph.org/), [Project Freya PID Graph](https://www.project-freya.eu/en/blogs/blogs/the-pid-graph) and the [Open Research Knowledge Graph](https://projects.tib.eu/orkg).
