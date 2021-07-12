import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, AutoTokenizer
from collections import Counter
import os
import shap
import scipy as sp
import matplotlib.pyplot as plt 
import socket

#ip_addr = 'localhost:5555'

ip_addr = str(socket.gethostbyname(socket.gethostname()))+':5555'


# df = pd.read_csv('./CAD_test10_preprocessed.csv', encoding='utf-8')
# test_text = df['clinical_diagnosis_conclusion']
device = torch.device("cuda")
bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
class BERT_Arch(nn.Module):
        def __init__(self, bert):
            super(BERT_Arch, self).__init__()
            self.bert = bert 
        # dropout layer
            self.dropout = nn.Dropout(0.5)
        
        # relu activation function
            self.relu =  nn.ReLU()

        # dense layer 1
            self.fc1 = nn.Linear(768,512)
        
        # dense layer 2 (Output layer)
            self.fc2 = nn.Linear(512,2)

        #softmax activation function
            self.softmax = nn.LogSoftmax(dim=1)

        #define the forward pass
        def forward(self, sent_id, mask):

        #pass the inputs to the model  
            _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
            
            x = self.fc1(cls_hs)
            x = self.relu(x)
        # output layer
            x = self.fc2(x)
        # apply softmax activation
            x = self.softmax(x)
            
            #Bert2
            return x

model = BERT_Arch(bert) 
m = torch.load('./saved_weights_clincal_diagnosis_weight14_0609data.pt')
model.load_state_dict(m)
model = model.to(device)

def predict():
    df = pd.read_csv('./CAD_test10_preprocessed.csv', encoding='utf-8')
    test_text = df['clinical_diagnosis_conclusion']
    # tokenize and encode sequences in the test set
    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(),
        max_length = 512,
        padding=True,
        truncation=True
    )
    from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

    #define a batch size
    batch_size = 1
    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    #test_id = torch.tensor(test_id_series.tolist())

    test_data = TensorDataset(test_seq, test_mask)

    # sampler for sampling the data during training
    test_sampler = SequentialSampler(test_data)

    # dataLoader for validation set
    test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size=batch_size)

    total_preds={}
    total_probs={}
    for step, batch in enumerate(test_dataloader):
            # push the batch to gpu
            batch = [t.to(device) for t in batch]
            sent_id, mask = batch
            
            # deactivate autograd
            with torch.no_grad():
        
                # model predictions
                preds = model(sent_id, mask)

                # compute the validation loss between actual and predicted values
                #loss = cross_entropy(preds,labels)
                #total_loss = total_loss + loss.item()
                preds = preds.detach().cpu().numpy()
                probs = np.exp(preds[:,1])
                preds = np.argmax(preds, axis = 1)
                total_preds[df['id'][step]] = preds[0]
                total_probs[df['id'][step]] = probs[0]
    # print(total_probs)
    # print(total_preds)
    return total_probs


def Shap_text():
    df = pd.read_csv('./CAD_test10_preprocessed.csv', encoding='utf-8')
    test_text = df['clinical_diagnosis_conclusion']
    text_lists = test_text
    def f(x):
        seq_list = []
        mask_list = []
        for v in x:
            tokens_train = tokenizer.encode_plus(v, padding='max_length', max_length=512, truncation=True)
            seq_list.append(tokens_train['input_ids'])
            mask_list.append(tokens_train['attention_mask'])
            #train_y = torch.tensor(test_labels.tolist())
        seq_list = torch.tensor(seq_list)
        #print(seq_list)
        mask_list = torch.tensor(mask_list)

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(seq_list.to(device), mask_list.to(device))
            # compute the validation loss between actual and predicted values
            preds = preds.detach().cpu().numpy()
            outputs = preds
            scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
            val = sp.special.logit(scores[:,1]) # use one vs rest logit units
            return val


    # build an explainer using a token masker
    explainer = shap.Explainer(f, tokenizer)

    shap_values = explainer(text_lists, fixed_context=1)
    #print(shap_values[1])
    #print(shap_values)
    shap.initjs()
    directory = {}
    date = {}
    # IPython.display.DisplayObject(shap.plots.text(shap_values))
    for i in range(len(df)):
        shap_text = shap.plots.text(shap_values[i])
        patient_format = str(df['operation day'][i]) + str(df['id'][i])
        with open('./app/templates/conclu{}.html'.format(patient_format), 'w') as f:
            f.write(shap_text)
        directory[df['id'][i]] = 'conclu{}.html'.format(patient_format)
        directory[df['id'][i]] = '{}/shap/conclu?id={}&date={}'.format(ip_addr, df['id'][i], df['operation day'][i])
        date[df['id'][i]] = df['operation day'][i]
    return directory, date
        
    # shap_text = shap.force_plot(shap_values[0])
    # shap.save_html("explainer.html", shap_text)

    #shap.plots.bar(shap_values.max(0), max_display=20, show=False)

    # #plt.show()
    #plt.savefig('bar_chart{}'.format(patient_format))
    # plt.close()
