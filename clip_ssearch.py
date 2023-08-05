import sys
import os
import clip
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
#Please, change the following path to where convnet2 can be located
sys.path.append("..\convnet2")
import datasets.data as data
import utils.configuration as conf
import utils.imgproc as imgproc
import skimage.transform as trans
from scipy import spatial

class CLIPSSearch:
    def __init__(self, config_file, model_name, checkpoint_path=None):
        
        self.configuration = conf.ConfigurationFile(config_file, model_name)
        #defiing input_shape                    
        self.input_shape =  (self.configuration.get_image_height(), 
                             self.configuration.get_image_width(),
                             self.configuration.get_number_of_channels())                       
        #loading the model
        if checkpoint_path is None:
            ############ HUGGING FACE ###############
            #model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            #processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            ############## PYTORCH ################
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, processor = clip.load("ViT-B/32", device=device)
            ######################################
        else:
            ############## HUGGING FACE ##################
            #model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            #processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            ######################################

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, processor = clip.load("ViT-B/32", device=device)
            model.load_state_dict(torch.load(checkpoint_path))
        self.device = device

        #redefining the model to get the hisdden output
        #self.output_layer_name = 'conv4_block6_out'
 
        #defining image processing function
        #self.process_fun =  imgproc.process_image_visual_attribute
        self.sim_model = model
        self.process_fun = processor
        
        #loading catalog
        self.ssearch_dir = os.path.join(self.configuration.get_data_dir(), 'ssearch')
        catalog_file = os.path.join(self.ssearch_dir, 'catalog.txt')        
        assert os.path.exists(catalog_file), '{} does not exist'.format(catalog_file)
        print('loading catalog ...')
        self.load_catalog(catalog_file)
        print('loading catalog ok ...')
        self.enable_search = False

    def normalize(self, data) :
        """
        unit normalization
        """
        norm = np.sqrt(np.sum(np.square(data), axis = 1))
        norm = np.expand_dims(norm, 0)  
        #print(norm)      
        data = data / np.transpose(norm)
        return data

    def square_root_norm(self, data) :
        return self.normalize(np.sign(data)*np.sqrt(np.abs(data)))

    def read_image(self, filename):      
        #print("Reading {}".format(filename))  
        image = Image.open(filename)
        #print(image)
        #print(type(image))
        try:
            ########### HUGGING FACE ################
            #im = self.process_fun(images=image, return_tensors="np")
            ########### PYTORCH ################
            im = self.process_fun(image).unsqueeze(0).to(self.device)
        except:
            image = image.convert('RGB')
            ############ HUGGING FACE #############
            #im = self.process_fun(images=image, return_tensors="np")
            ############ PYTORCH #############
            im = self.process_fun(image).unsqueeze(0).to(self.device)
        return im
    
    def read_image_2(self, filename):      
        #print("Reading {}".format(filename))  
        im = self.process_fun(data.read_image(filename, self.input_shape[2]), (self.input_shape[0], self.input_shape[1]))        
        #for resnet
        #im = tf.keras.applications.resnet50.preprocess_input(im)    
        return im
    
    def load_features(self):
        fvs_file = os.path.join(self.ssearch_dir, "features.np")                        
        fshape_file = os.path.join(self.ssearch_dir, "features_shape.np")
        features_shape = np.fromfile(fshape_file, dtype = np.int32)
        self.features = np.fromfile(fvs_file, dtype = np.float32)
        self.features = np.reshape(self.features, features_shape)
        self.enable_search = True
        print('features loaded ok')

    def load_catalog(self, catalog):
        with open(catalog) as f_in :
            self.filenames = [filename.replace('E:/', 'F:/').strip() for filename in f_in ]
        self.data_size = len(self.filenames) 

    def get_filenames(self, idxs):
        return [self.filenames[i] for i in idxs]

    def compute_features(self, image, expand_dims = False):
        #image['pixel_values'] = torch.tensor(np.array(image['pixel_values'])) # Its apparently faster
        with torch.no_grad():
            ############### HUGGING FACE ###############
            #fv = self.sim_model.get_image_features(**image)
            ############### PYTORCH ###############
            fv = self.sim_model.encode_image(image)
        fv = fv.cpu().numpy()[0]
        if expand_dims:
            fv = np.reshape(fv, (1, len(fv)))
        return fv

    def compute_features_from_catalog(self):
        result = []
        for i, filename in tqdm(enumerate(self.filenames)) :
            #print('reading {}: {}'.format(i, filename))
            result.append(self.compute_features(self.read_image(filename)))
        fvs = np.asarray(result)
        #fvs = np.concatenate(result)    
        print('fvs {}'.format(fvs.shape))
        fvs_file = os.path.join(self.ssearch_dir, "features.np")
        fshape_file = os.path.join(self.ssearch_dir, "features_shape.np")
        np.asarray(fvs.shape).astype(np.int32).tofile(fshape_file)       
        fvs.astype(np.float32).tofile(fvs_file)
        print('fvs saved at {}'.format(fvs_file))
        print('fshape saved at {}'.format(fshape_file))

    def adjust_query_embedding(self, query, original_embeddings, top=3, decide=True, df=None):
        data = self.features
        d = np.sqrt(np.sum(np.square(original_embeddings - query[0]), axis = 1))
        idx_sorted = np.argsort(d)
        visual_embeddings = data[idx_sorted[:top]]
        #visual_embeddings = np.vstack([visual_embeddings, query])
        #print(query.shape)
        new_query = np.mean(visual_embeddings, axis=0).reshape(1, len(query[0]))

        if decide:
            r_filenames = self.get_filenames(idx_sorted[:top])
            categories = []
            for i, file in enumerate(r_filenames):
                base = os.path.basename(file)
                filename = os.path.splitext(base)[0]
                name_and_productid = filename.rsplit('_', 1)
                try:
                    category = df.loc[(df['Title'] == name_and_productid[0]) & (df['ProductId'] == int(name_and_productid[1])), "GlobalCategoryEN"].values[0]
                except:
                    category = df.loc[(df['Title'] == name_and_productid[0]) & (df['ProductId'] == name_and_productid[1]), "GlobalCategoryEN"].values[0]
                categories.append(category)
            
            adjust = all(x == categories[0] for x in categories)
            if adjust:
                return new_query
            else:
                return query

        return new_query

    def adjust_query_embedding_sim(self, query, original_embeddings, text_model, top=3, decide=True, df=None):
        data = self.features
        d = np.sqrt(np.sum(np.square(original_embeddings - query[0]), axis = 1))
        idx_sorted = np.argsort(d)
        visual_embeddings = data[idx_sorted[:top]]
        #visual_embeddings = np.vstack([visual_embeddings, query])
        #print(query.shape)
        #print("len(query[0]): ", len(query[0]))
        new_query = np.mean(visual_embeddings, axis=0).reshape(1, len(query[0]))

        if decide:
            r_filenames = self.get_filenames(idx_sorted[:top])
            categories = []
            for i, file in enumerate(r_filenames):
                base = os.path.basename(file)
                filename = os.path.splitext(base)[0]
                name_and_productid = filename.rsplit('_', 1)
                try:
                    product_description = df.loc[(df['Title'] == name_and_productid[0]) & (df['ProductId'] == int(name_and_productid[1])), "ProductDescriptionEN"].values[0]
                except:
                    product_description = df.loc[(df['Title'] == name_and_productid[0]) & (df['ProductId'] == name_and_productid[1]), "ProductDescriptionEN"].values[0]
                #print("Description {}: {}".format(i, product_description))
                if text_model.model_name == "clip-base":
                    text_input = text_model.tokenizer(product_description, truncate=True).to(text_model.device)
                    with torch.no_grad():
                        text_features = text_model.model.encode_text(text_input)
                    text_features = text_features.cpu().numpy()[0]
                    data = text_features.astype(np.float32)
                    categories.append(data)
                elif text_model.model_name == "roberta":
                    data = text_model.model.encode(product_description)
                    categories.append(data)
                
            cos_sim_1 = 1 - spatial.distance.cosine(categories[0], categories[1])
            cos_sim_2 = 1 - spatial.distance.cosine(categories[0], categories[2])
            cos_sim_3 = 1 - spatial.distance.cosine(categories[1], categories[2])
            cos_sim_list = [cos_sim_1, cos_sim_2, cos_sim_3]
            adjust = all(cos_sim >= 0.8 for cos_sim in cos_sim_list)
            if adjust:
                return new_query
            else:
                return query
                #return new_query
        return new_query


    def search(self, im_query, metric = 'l2', norm = 'None', top=90, reducer=None, vtnn=None, adjust_query=False, adjust_query_sim=False, original_embeddings=None, df=None, text_model=None):
        assert self.enable_search, 'search is not allowed'
        q_fv = self.compute_features(im_query, expand_dims = True) # EXPAND DIMS ESTABA TRUE
        #print("features of im_query: {}".format(q_fv))
        if adjust_query:
            q_fv = self.adjust_query_embedding(query=q_fv, original_embeddings=original_embeddings, top=3, df=df)
            #print("features of im_query adjusted: {}".format(q_fv))
        if adjust_query_sim:
            q_fv = self.adjust_query_embedding_sim(query=q_fv, original_embeddings=original_embeddings, text_model=text_model, top=3, df=df)
        if vtnn is not None:
            q_fv = torch.tensor(q_fv)
            vtnn.eval()
            with torch.no_grad():
                q_fv = q_fv.to('cuda')
                q_fv = q_fv.view(-1, 2048)
                q_fv = vtnn(q_fv).cpu().numpy()
        #print("EMBEDDING SIZE: {}".format(len(q_fv[0])))
        #it seems that Euclidean performs better than cosine
        if metric == 'l2':
            if reducer is not None:
                data = reducer.transform(self.features)
                query = reducer.transform(q_fv)
            else:
                data = self.features
                query = q_fv
            if norm == 'square_root':
                data = self.square_root_norm(data)
                query = self.square_root_norm(query)
            #print("Query features:", query.shape)
            #print("data features:", data.shape)
            d = np.sqrt(np.sum(np.square(data - query[0]), axis = 1))
            #print("Distances unsorted: {}".format(d))
            idx_sorted = np.argsort(d)
            #print("idx sorted: {}".format(idx_sorted[:top]))
            d_sorted = np.sort(d)
            #print("Distances sorted: {}".format(d_sorted[:top]))
            #print(d[idx_sorted][:20])
        elif metric == 'cos' : 
            if norm == 'square_root':
                self.features = self.square_root_norm(self.features)
                q_fv = self.square_root_norm(q_fv)
            sim = np.matmul(self.normalize(self.features), np.transpose(self.normalize(q_fv)))
            sim = np.reshape(sim, (-1))            
            idx_sorted = np.argsort(-sim)
            d_sorted = -np.sort(-sim)
            #print(sim[idx_sorted][:20])
        #print("idx_sorted: ", idx_sorted[:top])

        if top is not None:
            return idx_sorted[:top], d_sorted[:top], q_fv, data
        return idx_sorted, d_sorted, q_fv, data

    def draw_result(self, filenames, similarity=None, distance=None):
        w = 1000
        h = 1000
        #w_i = np.int(w / 10)
        w_i = int(w / 10)
        #h_i = np.int(h / 10)
        h_i = int(h / 10)
        image_r = np.zeros((w,h,3), dtype = np.uint8) + 255
        x = 0
        y = 0
        for i, filename in enumerate(filenames) :
            pos = (i * w_i)
            x = pos % w
            #y = np.int(np.floor(pos / w)) * h_i
            y = int(np.floor(pos / w)) * h_i
            image = data.read_image(filename, 3)
            
            '''### Add text with the product id
            try:
                base = os.path.basename(filename)
                filename = os.path.splitext(base)[0]
                name_and_productid = filename.rsplit('_', 1)
                font = ImageFont.truetype("arial.ttf", 30)
                PIL_image = Image.fromarray(np.uint8(image)).convert('RGB')
                draw = ImageDraw.Draw(PIL_image)
                if (similarity is None and distance is None) or (i == 0):
                    draw.text((0, 0), "id: {}".format(name_and_productid[1]), font=font, fill='rgb(0, 0, 0)')
                elif similarity is not None:
                    draw.text((0, 0), "id: {} / sim: {}".format(name_and_productid[1], round(similarity[i - 1], 4)), font=font, fill='rgb(0, 0, 0)')
                elif distance is not None:
                    draw.text((0, 0), "id: {} / dist: {}".format(name_and_productid[1], round(distance[i - 1], 4)), font=font, fill='rgb(0, 0, 0)')
            except:
                #print("Could not write id for product.")
                pass'''
            PIL_image = Image.fromarray(np.uint8(image)).convert('RGB')

            image = np.array(PIL_image)        
            image = imgproc.toUINT8(trans.resize(image, (h_i,w_i)))
            image_r[y:y+h_i, x : x +  w_i, :] = image              
        return image_r   

if __name__ == '__main__':
    with open('F:\Documentos Universidad\MEMORIA\content_HomyCLIPBASE\ssearch\catalog.txt','r',encoding='utf-8') as file :
        lines = file.readlines()
        for line in lines:
            print(line)