import enum
import os
import copy
import clip
import gensim
import umap
import statistics
import pandas as pd
import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
from typing import OrderedDict
from operator import itemgetter
from ssearch import avg_precision as ssearch_avg_precision
from tqdm import tqdm
import scipy.spatial as sp
import statistics
from sentence_transformers import  SentenceTransformer
import torch
from transformers import BertModel, BertTokenizer

class TextSearch:
    def __init__(self, model_name="word2vec", filter_by_nwords=False, build=True, dataset="Pepeganga"):
        self.model = None
        self.model_name = model_name
        self.build = build
        self.filter = filter_by_nwords
        self.model_name=model_name
        ########## TESTING WITH CLIP VISUAL ENCODER AND ROBERTA ##########
        ##################################################################
        self.data_path = "./catalogues/{}/data/".format(dataset)
        self.q_file_path = self.data_path + "questions_dataset.xlsx"
        self.a_file_path = self.data_path + "answers_dataset.xlsx"
        self.visual_text_embeddings_path = "./visual_text_embeddings/{}/{}".format(dataset, self.model_name) 
        self.a_df = None
        self.q_df = None
        self.data = None
        self.sorted_similarity = None
        self.sorted_idx_similarity = None
        self.product_names = None
        self.dataset = dataset

    def set_model(self):
        if self.model_name == "mpnet":
            self.set_mpnet_model(str="all-mpnet-base-v2")
        elif self.model_name == "roberta":
            self.set_roberta_model(str="all-roberta-large-v1")
        elif self.model_name == "bert-cased-768" or self.model_name == "bert-cased-3072":
            self.set_cased_bert_model()
        elif self.model_name == "bert-uncased-768":
            self.set_uncased_bert_model()
        elif self.model_name == "word2vec":
            self.set_gensim_model(str="word2vec-google-news-300")
        elif self.model_name == "clip-base":
            self.set_clip_model()
        elif self.model_name == "clip-fn":
            self.set_clipfn_model()
        else:
            raise ValueError('Model specified is not valid.')

    def set_clip_model(self):
        if self.build:
            ############ HUGGING FACE ##################
            #clip_config = CLIPConfig(text_config_dict={"max_position_embeddings":1024})
            #model = CLIPModel(config=clip_config)#.from_pretrained("openai/clip-vit-base-patch32")
            #tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            #self.tokenizer = tokenizer
            ############ PYTORCH ##################
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _ = clip.load("ViT-B/32", device=device)
            self.model = model
            self.tokenizer = clip.tokenize
            self.device = device
        else:
            print("Build is set to False. No reason to set a model.")
    
    def set_clipfn_model(self):
        if self.build:
            ############ HUGGING FACE ##################
            #clip_config = CLIPConfig(text_config_dict={"max_position_embeddings":1024})
            #model = CLIPModel(config=clip_config).from_pretrained("F:/Documentos Universidad\MEMORIA\CLIP_checkpoints\model3_lr0.001.pt")
            #tokenizer = CLIPTokenizer.from_pretrained("F:/Documentos Universidad\MEMORIA\CLIP_checkpoints\model3_lr0.001.pt")
            #self.tokenizer = tokenizer
            ############ PYTORCH ##################
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _ = clip.load("ViT-B/32", device=device)
            model.load_state_dict(torch.load("F:/Documentos Universidad\MEMORIA\CLIP_models/{}/model1.pt".format(self.dataset)))
            self.model = model
            self.tokenizer = clip.tokenize
            self.device = device
        else:
            print("Build is set to False. No reason to set a model.")

    def set_cased_bert_model(self):
        if self.build:
            model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.model = model
            self.tokenizer = tokenizer
        else:
            print("Build is set to False. No reason to set a model.")

    def set_uncased_bert_model(self):
        if self.build:
            model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = model
            self.tokenizer = tokenizer
        else:
            print("Build is set to False. No reason to set a model.")

    def set_gensim_model(self, str="word2vec-google-news-300"):
        if self.build:
            self.model = api.load(str)
        else:
            print("Build is set to False. No reason to set a model.")

    def set_roberta_model(self, str="all-roberta-large-v1"):
        if self.build:
            self.model = SentenceTransformer(str)
        else:
            print("Build is set to False. No reason to set a model.")

    def set_mpnet_model(self, str="all-mpnet-base-v2"):
        if self.build:
            self.model = SentenceTransformer(str)
        else:
            print("Build is set to False. No reason to set a model.")

    def get_model_name(self):
        return self.model_name

    def check_df(self, filenames):
        a_df = pd.read_pickle(self.data_path + "answers_dataset_vector_old.pkl")
        print("Largo de filenames: ", len(filenames))
        print("Before: ", len(a_df))
        aux_filenames = np.array([os.path.splitext(os.path.basename(file))[0] for file in filenames])
        for index, row in a_df.iterrows():
            if row['name_and_productid'] not in aux_filenames:
                a_df.drop(index, inplace=True)
        
        a_df.to_pickle(self.data_path + "answers_dataset_vector.pkl")
        return a_df

    def set_dataframes(self):
        if self.build:
        #if False:
            questions = pd.read_excel(self.q_file_path)
            answers = pd.read_excel(self.a_file_path)
            q_df, a_df, a_data, q_data, product_names, q_product_names = self.build_dataframes(questions, answers, self.filter)
        else:
            q_df = pd.read_pickle(self.data_path + "{}/questions_dataset_vector.pkl".format(self.model_name))
            a_df = pd.read_pickle(self.data_path + "{}/answers_dataset_vector.pkl".format(self.model_name))
            a_data = np.stack(a_df['descVector'].to_numpy())
            q_data = np.stack(q_df['descVector'].to_numpy())
            product_names = a_df['name_and_productid'].to_numpy()
            q_product_names = q_df['name_and_productid'].to_numpy()
        self.a_df = a_df
        self.q_df = q_df
        self.data = a_data
        self.q_data = q_data
        self.product_names = product_names
        self.q_product_names = q_product_names

    def build_dataframes(self, questions_df, answers_df, filter=False, n_words=30):
        q_df = questions_df.dropna(subset=["ProductDescriptionEN"])
        a_df = answers_df.dropna(subset=["ProductDescriptionEN"])

        if filter:
            wantedRows = a_df[a_df['ProductDescriptionEN'].str.split().str.len()<n_words].index
            a_df = a_df.drop(wantedRows, axis=0)

        tqdm.pandas()
        print("Building questions dataframe...")
        q_df['descVector'] = q_df.progress_apply(get_desc_vector, text_model=self, axis=1)
        print("Building answers dataframe...")
        a_df['descVector'] = a_df.progress_apply(get_desc_vector, text_model=self, axis=1)

        q_df['name_and_productid'] = q_df.apply(generate_name_and_productid, axis=1)
        a_df['name_and_productid'] = a_df.apply(generate_name_and_productid, axis=1)

        print("Saving dataframes with vectors")
        q_df.to_pickle(self.data_path + "{}/questions_dataset_vector.pkl".format(self.model_name))
        a_df.to_pickle(self.data_path + "{}/answers_dataset_vector.pkl".format(self.model_name))
        print("Done")

        data = np.stack(a_df['descVector'].to_numpy())
        q_data = np.stack(q_df['descVector'].to_numpy())
        product_names = a_df['name_and_productid'].to_numpy()
        q_product_names = q_df['name_and_productid'].to_numpy()
        #print("product_names")
        #print(product_names)


        return q_df, a_df, data, q_data, product_names, q_product_names

    def cosine_sim(self):
        data = self.data.astype(np.float32)
        #data = self.data
        print("Data shape: ", np.shape(data))
        # se calcula norma l2
        norm = np.expand_dims(np.linalg.norm(self.data, axis=1), axis=1)

        # se divide por la norma
        data = data / norm

        # se calcula el producto punto entre todos los vectores
        sim_cos = np.matmul(data, np.transpose(data))
        print("Sim_cos shape: ", np.shape(sim_cos))

        #ahora obtemos los indices por fila, dado un orden descendente (ya que es similitud)
        idx_sorted = np.argsort(-sim_cos, axis=1)
        sim_cos_sorted = -np.sort(-sim_cos, axis=1)
        print("sorted_data")
        print(idx_sorted)
        self.sorted_similarity = sim_cos_sorted
        self.sorted_idx_similarity = idx_sorted

    def get_product_vector(self, product, answers=False):
        if answers:
            file_path = self.a_file_path
        else:
            file_path = self.q_file_path
        
        df = pd.read_excel(file_path)

        product_row = df.loc[df["Title"] == product]
        product_row['descVector'] = product_row.apply(get_desc_vector, model=self.model, axis=1)
        return product_row['descVector']
    
    def adjust_visual_embeddings(self, embeddings, filenames, k=3, method='mean', a=0.9, T=1.0, use_query=True):
        new_embeddings = []
        
        embeddings_files = [f for f in os.listdir(self.visual_text_embeddings_path)]
        if method == 'mean':
            if use_query:    
                generated_file = "k_{}_a_{}_{}.txt".format(str(k), str(a).replace(".", ""), method)
            else:
                generated_file = "k_{}_a_{}_{}_noquery.txt".format(str(k), str(a).replace(".", ""), method)
        elif method == 'sim':
            use_query = True
            generated_file = "k_{}_{}.txt".format(str(k), method)
        elif method == 'softmax':
            use_query = True
            generated_file = "k_{}_t_{}_{}.txt".format(str(k), T, method)

        print("embeddings_files: ", embeddings_files)
        print("generated_file: ", generated_file)
        if generated_file in embeddings_files:
            print("Found file with adjusted embeddings. Loading...")
            new_embeddings = np.loadtxt("{}/{}".format(self.visual_text_embeddings_path, generated_file), delimiter='\t')
            return new_embeddings
        
        ################
        #self.cosine_sim()
        ################

        aux_filenames = np.array([os.path.splitext(os.path.basename(file))[0] for file in filenames])
        for id, embedding in enumerate(tqdm(embeddings)):
            product_name = os.path.splitext(os.path.basename(filenames[id]))[0]
            try:
                text_idx = np.where(self.product_names == product_name)[0][0]
            except:
                print("Product {} has a problem. Setting original embedding.".format(product_name))
                new_embeddings.append(embedding)
                continue

            # top-k ids de los productos ordenados por similitud al producto base
            all_neighbors_idx = self.sorted_idx_similarity[text_idx]
            all_neighbors_idx = np.delete(all_neighbors_idx, np.where(all_neighbors_idx == text_idx))
            if use_query or method == 'sim' or method == 'softmax':
                all_neighbors_idx = np.insert(all_neighbors_idx, 0, text_idx)
            neighbors_idx = all_neighbors_idx[:k]

            if text_idx not in neighbors_idx and use_query:
                print("Wrong. Text_id not in neighbors_idx when it should")

            elif text_idx in neighbors_idx and not use_query:
                print("Wrong. Text_id in neighbors_idx when it shouldnt")

            # obtenemos los nombres de los productos
            neighbors_names = self.product_names[neighbors_idx]
            #print("Neighbours by text: {}".format(neighbors_names))
            
            # obtenemos los indices de los productos que se usaran para generar el nuevo embedding 
            visual_idx = np.where(np.in1d(aux_filenames, neighbors_names))[0]

            # obtenemos los embeddings visuales
            visual_embeddings = embeddings[visual_idx]

            # calculamos el promedio
            if method == 'mean':
                np_mean = np.mean(visual_embeddings, axis=0)
                mean_vector = np_mean.tolist()

                base_embedding = a * np.array(embedding)
                adjust_embedding = (1 - a) * np.array(mean_vector)
                new_embedding = base_embedding + adjust_embedding
            
            elif method == 'sim':
                weights = self.sorted_similarity[text_idx][:k]
                weights_sum = np.sum(weights)
                weights_normalized = weights/weights_sum
                new_embedding = np.average(visual_embeddings, weights=weights_normalized, axis=0)

            elif method == 'softmax':
                cos_sim = self.sorted_similarity[text_idx][:k]
                weights = t_softmax(cos_sim, T=T)
                new_embedding = np.average(visual_embeddings, weights=weights, axis=0)
                #print("Old embedding: {}".format(embedding))
                #print("New embedding: {}".format(new_embedding))

            new_embeddings.append(new_embedding)

        np.savetxt("{}/{}".format(self.visual_text_embeddings_path, generated_file), np.array(new_embeddings), delimiter='\t')

        return new_embeddings

    def search_product(self, product, k=20):
        q_file_path = self.q_file_path
        a_file_path = self.a_file_path

        print("Reading questions and answers files defined in init")
        questions = pd.read_excel(q_file_path)
        #print(questions)
        answers = pd.read_excel(a_file_path)

        product = questions.loc[questions["Title"] == product]
        #product['descVector'] = product.apply(get_desc_vector, model=self.model, axis=1)
        q_df, a_df = self.build_dataframes(product, answers, self.filter)

        ap_arr, res = self.get_results(q_df, a_df, k)

        return ap_arr, statistics.mean(ap_arr), res

    def search_product_by_name(self, name, k=99):
        self.set_dataframes()
        name_and_productid = name.rsplit('_', 1)
        desc_vector = self.a_df.loc[(self.a_df['Title'] == name_and_productid[0]) & (self.a_df['ProductId'] == int(name_and_productid[1])), "descVector"].values[0]
        q_data = np.reshape(desc_vector, (1, len(desc_vector)))
        data = self.data

        print("Answers Data shape: ", np.shape(data))
        print("Questions Data shape: ", np.shape(q_data))
        
        sim_cos = 1 - sp.distance.cdist(q_data, data, 'cosine')
        print("Sim_cos shape: ", np.shape(sim_cos))
        #print("Sim_cos: ", sim_cos)
        #ahora obtemos los indices por fila, dado un orden descendente (ya que es similitud)
        idx_sorted = np.argsort(-sim_cos, axis=1)
        sim_cos_sorted = -np.sort(-sim_cos, axis=1)

        a_df = self.a_df
        q_df = self.q_df


        #for q_idx, q_product in enumerate(self.q_product_names):
            #print("Query: {}".format(q_product))
            #base_categories = q_df.loc[q_df['name_and_productid'] == q_product, ["GlobalCategoryEN", "CategoryTree", "SubCategory"]].values[0].tolist()
        similar_products_idxs = idx_sorted[0][:k]
        similar_products_cossim = sim_cos_sorted[0][:k]
        similar_products_names = self.product_names[similar_products_idxs]
        return similar_products_names, similar_products_cossim

        #print("Query {}: {}\nResults: {}\nSimilarities: {}\n\n".format(q_idx, q_product, similar_products_names, similar_products_cossim))

    def eval_model(self, k=20, use_umap=False, n_components=None, seed=42):

        self.set_dataframes()
        #self.cosine_sim()

        #ap_arr, res = self.get_results2(q_df, a_df, k)
        q_data = self.q_data
        data = self.data

        if use_umap:
            reducer = umap.UMAP(n_epochs=500, n_components=n_components, random_state=seed)
            reducer.fit(data)
            data = reducer.transform(data)
            q_data = reducer.transform(q_data)

        print("Answers Data shape: ", np.shape(data))
        print("Questions Data shape: ", np.shape(q_data))
        
        sim_cos = 1 - sp.distance.cdist(q_data, data, 'cosine')
        print("Sim_cos shape: ", np.shape(sim_cos))
        #print("Sim_cos: ", sim_cos)
        #ahora obtemos los indices por fila, dado un orden descendente (ya que es similitud)
        idx_sorted = np.argsort(-sim_cos, axis=1)
        sim_cos_sorted = -np.sort(-sim_cos, axis=1)

        a_df = self.a_df
        q_df = self.q_df

        ap_arr = []
        ap_arr_tree = []
        ap_arr_sub = []
        
        for q_idx, q_product in enumerate(self.q_product_names):
            #print("Query: {}".format(q_product))
            base_categories = q_df.loc[q_df['name_and_productid'] == q_product, ["GlobalCategoryEN", "CategoryTree", "SubCategory"]].values[0].tolist()
            similar_products_idxs = idx_sorted[q_idx][:k]
            similar_products_cossim = sim_cos_sorted[q_idx][:k]
            similar_products_names = self.product_names[similar_products_idxs]
            similar_products_categories = []
            #print("Query {}: {}\nResults: {}\nSimilarities: {}\n\n".format(q_idx, q_product, similar_products_names, similar_products_cossim))
            for a_product in similar_products_names:
                categories = a_df.loc[a_df['name_and_productid'] == a_product, ["GlobalCategoryEN", "CategoryTree", "SubCategory"]].values[0].tolist()
                similar_products_categories.append(categories)
            ap, ap_tree, ap_sub = avg_precision(base_categories, similar_products_categories, use_all_categories=True)
            ap_arr.append(ap)
            ap_arr_tree.append(ap_tree)
            ap_arr_sub.append(ap_sub)

        #print(sim_cos_sorted)
        #print(idx_sorted)

        mAP = statistics.mean(ap_arr)
        mAP_tree = statistics.mean(ap_arr_tree)
        mAP_sub = statistics.mean(ap_arr_sub)

        return mAP, mAP_tree, mAP_sub

    def get_results(self, q_df, a_df, k, test=False):
        ap_arr = []
        res = dict()
        # For testing purposes
        iteration = 0
        ######################
        print("Iterating through rows")
        for index, q_row in q_df.iterrows():
            iteration += 1
            print("Iteration {}".format(iteration))

            similarity_arr = []
            q_vec = q_row["descVector"]
            q_text = q_row["ProductDescriptionEN"]
            q_category = q_row["GlobalCategoryEN"]
            q_product = q_row["Title"]
            q_img_url = q_row["ImageUrl"]

            #if use_umap:
            #    q_vec = reducer.transform(q_vec.reshape(1, -1))[0]

            res['Query Product {}'.format(index)] = dict()
            res['Query Product {}'.format(index)]['Title'] = q_product
            res['Query Product {}'.format(index)]['Global Category'] = q_category
            res['Query Product {}'.format(index)]['Description'] = q_text
            res['Query Product {}'.format(index)]['ImageUrl'] = q_img_url
            #res['Query Product {}'.format(index)]['Results'] = dict()
            answers_dict = dict()

            for i, a_row in a_df.iterrows():
                a_vec = a_row["descVector"]
                a_text = a_row["ProductDescriptionEN"]
                a_category = a_row["GlobalCategoryEN"]
                a_product = a_row["Title"]
                a_img_url = a_row["ImageUrl"]
                #print("Answer text: {}".format(a_text))
                
                #if use_umap:
                #    a_vec = reducer.transform(a_vec.reshape(1, -1))[0]

                similarity = cosine_similarity(q_vec.reshape(1, -1), a_vec.reshape(1, -1))
                this_similarity = (a_text, a_category, similarity, a_product)
                similarity_arr.append(this_similarity)

                answers_dict[a_product] = dict()
                answers_dict[a_product]['Title'] = a_product
                answers_dict[a_product]['Global Category'] = a_category
                answers_dict[a_product]['Description'] = a_text
                answers_dict[a_product]['Similarity'] = str(similarity[0][0])
                answers_dict[a_product]['ImageUrl'] = a_img_url

            sorted_similarity = sorted(similarity_arr, key=itemgetter(2), reverse=True)
            maximums = sorted_similarity[:k]
            ap = ssearch_avg_precision(q_category, maximums)
            ap_arr.append(ap)
            res['Query Product {}'.format(index)]['Results'] = add_best_results(answers_dict, maximums, q_product)
            if iteration == 10 and test:
                print("Retrieving only 10 products")
                break
        return ap_arr, res


def get_desc_vector(row, text_model):
    """Generates and returns a vector of the description of a product.

    Args:
        row (pandas.DataFrame row): Individual row of a dataframe containing the description of a 
            specific product.
        model (gensim model): Model used to generate the word vectors.
    """
    if text_model.model_name == 'mpnet' or text_model.model_name == 'roberta':
        desc = row['ProductDescriptionEN']
        desc_vector = text_model.model.encode(desc)
        return desc_vector

    elif text_model.model_name == "bert-cased-768" or text_model.model_name == "bert-uncased-768":
        desc = row['ProductDescriptionEN']
        desc_id = text_model.tokenizer.encode(desc, truncation=True)
        desc_id = torch.LongTensor(desc_id)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        bert_model = text_model.model.to(device)
        desc_id = desc_id.to(device)
        bert_model.eval()
        desc_id = desc_id.unsqueeze(0)
        with torch.no_grad():
            out = bert_model(input_ids=desc_id)
        # we only want the hidden_states
        hidden_states = out[2]
        sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()
        return sentence_embedding.cpu().numpy()

    elif text_model.model_name == "bert-cased-3072":
        desc = row['ProductDescriptionEN']
        desc_id = text_model.tokenizer.encode(desc, truncation=True)
        desc_id = torch.LongTensor(desc_id)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        bert_model = text_model.model.to(device)
        desc_id = desc_id.to(device)
        bert_model.eval()
        desc_id = desc_id.unsqueeze(0)
        with torch.no_grad():
            out = bert_model(input_ids=desc_id)
        # we only want the hidden_states
        hidden_states = out[2]
        last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
        cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
        cat_sentence_embedding = torch.mean(cat_hidden_states, dim=1).squeeze()
        return cat_sentence_embedding.numpy()

    elif text_model.model_name == "clip-base" or text_model.model_name == "clip-fn":
        desc = row['ProductDescriptionEN']
        ################# HUGGING FACE ################
        #text_input = text_model.tokenizer(desc, padding=True, return_tensors="pt")
        #with torch.no_grad():
        #    text_features = text_model.model.get_text_features(**text_input)
        ################# PYTORCH ################
        text_input = text_model.tokenizer(desc, truncate=True).to(text_model.device)
        with torch.no_grad():
            text_features = text_model.model.encode_text(text_input)
        return text_features.cpu().numpy()[0]
    else:
        words = gensim.utils.simple_preprocess(row['ProductDescriptionEN'])
        desc_vector = avg_sentence_vector(words=words, model=text_model.model, num_features=300)
        #print("desc_vector: ", desc_vector)
        return desc_vector

def generate_name_and_productid(row):
    name = row['Title']
    productid = row['ProductId']
    name_and_productid = name + "_" + str(productid)
    #print("desc_vector: ", desc_vector)
    return name_and_productid

def avg_sentence_vector(words, model, num_features):
    """Calculates and returns the average vector for the words given.

    Args:
        words (array(str)): array containing words. 
        model (gensim model): model used to generate the words vectors.
        num_features (int): Number of dimensions of the vector.
    """
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    for word in words:
        if word not in gensim.parsing.preprocessing.STOPWORDS:
            nwords = nwords+1
            try:    
                featureVec = np.add(featureVec, model[word])
            except:
                #print('Ignoring word: {} (not found)'.format(word))
                nwords -= 1
                continue
    #print("nwords: {}".format(nwords))
    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec


def add_best_results(d, maximums, q_product):
    """Generates and returns an ordered dict with the best results (by similarity)
    of a product.

    Args:
        d (dict): Unordered dictionay.
        maximums (array): array with the 'k' best results for 'q_product'
        q_product (str): Name of the product for which the best results are added.

    """
    print("Adding best results for {}".format(q_product))
    d2 = copy.deepcopy(d)
    for k in d2:
        tmp = [item for item in maximums if item[3] == k]
        if len(tmp) == 0:
            d.pop(k)

    ordered = OrderedDict(sorted(d.items(), key=lambda x: x[1]['Similarity'], reverse=True))
    #return d
    return ordered

def t_softmax(x, T=1.0):
    return np.exp(x / T) / np.sum(np.exp(x / T))

def avg_precision(y, y_pred, use_all_categories=False):
    p = 0
    n_relevant = 0
    pos = 1

    if use_all_categories:
        p_tree = 0
        n_relevant_tree = 0
        pos_tree = 1
        p_sub = 0
        n_relevant_sub = 0
        pos_sub = 1
        for product in y_pred:
            if product[0] == y[0]:
                n_relevant += 1
                p += n_relevant / pos
            pos += 1
            if product[1] == y[1]:
                n_relevant_tree += 1
                p_tree += n_relevant_tree / pos_tree
            pos_tree += 1
            if product[2] == y[2]:
                n_relevant_sub += 1
                p_sub += n_relevant_sub / pos_sub
            pos_sub += 1
        
        if n_relevant != 0:
            ap = p / n_relevant
        else:
            ap = 0 

        if n_relevant_tree != 0:
            ap_tree = p_tree / n_relevant_tree
        else:
            ap_tree = 0

        if n_relevant_sub != 0:
            ap_sub = p_sub / n_relevant_sub
        else:
            ap_sub = 0 

        return ap, ap_tree, ap_sub
    
    else:
        for product in y_pred:
            if product == y:
                n_relevant += 1
                p += n_relevant / pos
            pos += 1
    
        if n_relevant != 0:
            ap = p / n_relevant
        else:
            ap = 0 

        return ap

