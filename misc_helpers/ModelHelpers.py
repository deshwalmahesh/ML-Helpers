import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score,accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.metrics import top_k_categorical_accuracy, TopKCategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from transformers import AdamW
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.special import softmax
import joblib
from tqdm.notebook import tqdm
import tensorflow as tf
import datetime


SEED = 13
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.style.use('seaborn-bright')


# tokenize and encode sequences in the text list
def textToTensor(text,labels=None,paddingLength=30):
  
  tokens = tokenizer.batch_encode_plus(text.tolist() if isinstance(text,np.ndarray) else text, 
                                       max_length=paddingLength, 
                                       padding='max_length', 
                                       truncation=True)
  
  text_seq = torch.tensor(tokens['input_ids'])
  text_mask = torch.tensor(tokens['attention_mask'])

  text_y = None
  if isinstance(labels,np.ndarray): # if we do not have y values
    text_y = torch.tensor(labels.tolist() if isinstance(labels,np.ndarray) else labels)

  return text_seq, text_mask, text_y



def analyze_predictions(y_probs,y_true,labels_dict=None,force=False):
    if not force:
      assert y_probs.max()<1 and y_probs.min()>0, "Provide 'y_probs' as Softmax Probabilities or Confidence Scores, not Logits"
      
    y_pred = np.argmax(y_probs,axis=1)
    confidence = []
    for i,index in enumerate(y_pred):
        confidence.append(y_probs[i][index])

    df = pd.DataFrame({'y_true':y_true,'y_pred':y_pred,'confidence':confidence})
    df['match'] = df.apply(lambda row: True if row['y_true'] == row['y_pred'] else False ,axis=1)

    if labels_dict:
        df['y_true'].replace(labels_dict,inplace=True)
        df['y_pred'].replace(labels_dict,inplace=True)

    return df


def top_chapter_results(prob_row,label_to_index_mapping,chapter_to_class_mapping):
  five_preds = {}
  for index in prob_row:
    chapter = label_to_index_mapping[index] # single chapter name
    class_sub = chapter_to_class_mapping[chapter]

    five_preds[chapter] = class_sub
  return five_preds


def get_top_k_accuracy(y_probs,test_labels,k=5):
  metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=k)
  metric.update_state(np.expand_dims(test_labels,axis=1),y_probs)
  return metric.result().numpy()


class BERT_Subject_Classifier(nn.Module):

    def __init__(self,out_classes,hidden1=128,hidden2=32,dropout_val=0.2,unfreeze_n=79,logit=True):
      super(BERT_Subject_Classifier, self).__init__()

      assert ((type(unfreeze_n) == int) or (type(unfreeze_n) == str)), "unfreeze_n should be either 'none','all' or an int"

      self.hidden1 = hidden1
      self.hidden2 = hidden2
      self.dropout_val = dropout_val
      self.logits = logit
      self.bert = AutoModel.from_pretrained('bert-base-uncased')
      self.out_classes = out_classes
      self.unfreeze_n = unfreeze_n # make the last n layers trainable
      
      # dropout layer
      self.dropout = nn.Dropout(self.dropout_val)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,self.hidden1)
      self.fc2 = nn.Linear(self.hidden1,self.hidden2)
      
      # dense layer 2 (Output layer)
      self.fc3 = nn.Linear(self.hidden2,self.out_classes)

      #softmax activation function
      #self.softmax =  nn.LogSoftmax(dim=1)
      self.softmax = nn.Softmax(dim=1)


      if (type(self.unfreeze_n) == str and self.unfreeze_n == 'all') or (type(self.unfreeze_n) == int and self.unfreeze_n >= 199):
        for param in self.bert.parameters():
          param.requires_grad = True

      elif (type(self.unfreeze_n) == str and self.unfreeze_n == 'none') or (type(self.unfreeze_n) == int and self.unfreeze_n <= 0):
        for param in self.bert.parameters():
          param.requires_grad = False
      
      else:
        for param in list(self.bert.parameters())[:199-self.unfreeze_n]: # unfreeze n BERT Layers
          param.requires_grad = False
          

    #define the forward pass
    def forward(self, sent_id, mask):

      #pass the inputs to the model  
      _, cls_hs = self.bert(sent_id, attention_mask=mask)
      
      x = self.fc1(cls_hs)

      x = self.relu(x)

      x = self.dropout(x)

      # output layer
      x = self.fc2(x)

      x = self.dropout(x)

      x = self.fc3(x)

      if not self.logits:
        # apply softmax activation
        x = self.softmax(x)

      return x



class FocalLoss(nn.Module):
  def __init__(self,weight=None, alpha=1, gamma=2, logits=False, reduce=True):
    super(FocalLoss, self).__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.logits = logits
    self.reduce = reduce
    self.CLElossv = nn.CrossEntropyLoss(weight=weight)

  def forward(self, inputs, targets):
    BCE_loss = self.CLElossv(inputs, targets)

    pt = torch.exp(-BCE_loss)
    F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

    if self.reduce:
      return torch.mean(F_loss)
    else:
      return F_loss


# function to train the model
def train(model,optimizer,train_dataloader,loss_fun):
  
  model.train()

  total_loss, total_accuracy = 0, 0
  
  # empty list to save model predictions
  total_preds=[]
  
  # iterate over batches
  for step,batch in tqdm(enumerate(train_dataloader),desc="Train Steps"):
  
    # push the batch to gpu
    batch = [r.to(device) for r in batch]
 
    sent_id, mask, labels = batch

    # clear previously calculated gradients 
    model.zero_grad()        

    # get model predictions for the current batch
    preds = model(sent_id, mask)

    # compute the loss between actual and predicted values
    #loss = cross_entropy(preds, labels)
    loss = loss_fun(preds, labels)

    # add on to the total loss
    total_loss = total_loss + loss.item()

    # backward pass to calculate the gradients
    loss.backward()

    # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # update parameters
    optimizer.step()

    # model predictions are stored on GPU. So, push it to CPU
    preds=preds.detach().cpu().numpy()

    # append the model predictions
    total_preds.append(preds)

  # compute the training loss of the epoch
  avg_loss = total_loss / len(train_dataloader)
  
  # predictions are in the form of (no. of batches, size of batch, no. of classes).
  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  #returns the loss and predictions
  return avg_loss, total_preds


# function for testing or evaluating given y_labels
def evaluate_test(model,text=None,labels=None,dataloader=None,pad_len=30,batch_size=1024,return_predictions=False,loss_fun=None,mode='eval'):

  if mode != 'eval':
    seq,mask,_ = textToTensor(text,paddingLength=pad_len)
    data = TensorDataset(seq,mask)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

  model.eval()

  total_preds = []
  all_labels = []
  total_loss, avg_loss = 0, None

  for step,batch in tqdm(enumerate(dataloader),desc=f"{mode.capitalize()} Steps"):
    batch = [t.to(device) for t in batch]

    if mode=='eval':
      sent_id, mask, labels = batch
    else:
      sent_id, mask = batch

    with torch.no_grad():
      preds = model(sent_id, mask)

      if mode=='eval':
        loss = loss_fun(preds, labels)
        total_loss = total_loss + loss.item()
        all_labels.extend(labels.detach().cpu().numpy())

      total_preds.append(preds.detach().cpu().numpy()) # get the values in CPU after detatching


  model_pred_probs = np.concatenate(total_preds, axis=0) # probabilities of all the classes returned by model
  y_predicted = np.argmax(model_pred_probs,axis=1) # probabilities converted to classes by np.argmax()

  if mode == 'eval':
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels,y_predicted)
    f1 = f1_score(all_labels,y_predicted,average='weighted')
    return (avg_loss,acc,f1)
  
  if return_predictions:
    return model_pred_probs
  else: # concatenate all the prediction in the form of (number of samples, no. of classes) and then return a 1-D prediction array
    return y_predicted


def run_model_training(out_classes,optim=AdamW,pad_len=35,lr=1e-5,weight_decay=1e-4,hidden1=128,hidden2=32,dropout_val=0.20,batch_size=128,epochs=20,
                    save_best_weights=True,best_weight_name ='best_weight.pt',restore_best_weights=True,verbose=True,delta=0.001,patience=2,model_weights=None):
  
  best_valid_loss = float('inf')
  counter = 0

  # text2tensor
  train_seq,train_mask,train_y = textToTensor(train_text,train_labels,pad_len)
  val_seq,val_mask,val_y       = textToTensor(val_text,val_labels,pad_len)

  train_data = TensorDataset(train_seq, train_mask, train_y)
  train_sampler = RandomSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

  val_data = TensorDataset(val_seq, val_mask, val_y)
  val_sampler = SequentialSampler(val_data)
  val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=1024)

  model = BERT_Subject_Classifier(out_classes,hidden1,hidden2,dropout_val)
  model = model.to(device)
  if model_weights:
    tqdm.write('Existing Weights found. Changing Weights....')
    model.load_state_dict(torch.load(model_weights,map_location=device))

  optimizer = optim(model.parameters(),lr =lr, weight_decay=weight_decay)

  class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)
  weights = torch.tensor(class_weights,dtype=torch.float)
  weights = weights.to(device)
  focalLoss = FocalLoss(weight=weights)
  

  for epoch in tqdm(range(epochs),desc="Epochs"):
    train_loss, _ = train(model,optimizer,train_dataloader,loss_fun=focalLoss)
    valid_loss,val_acc,val_f1 = evaluate_test(model,dataloader=val_dataloader,loss_fun=focalLoss,mode='eval')
    if verbose:
      tqdm.write(f"End of Epoch:{epoch+1}\nloss:{train_loss:.3f}, val_loss:{valid_loss:3f},val_acc:{val_acc:3f},val_f1_w:{val_f1:3f}\n")
      
    #save the best model
    if valid_loss+delta < best_valid_loss and counter<patience :
      best_valid_loss = valid_loss
      counter=0
      if save_best_weights:
        if verbose:
          tqdm.write('val_loss improved. Saving weights...')
        torch.save(model.state_dict(),best_weight_name)
    else:
      counter+=1

    if counter == patience:
      break


  if restore_best_weights:
    model.load_state_dict(torch.load(best_weight_name))

  y_probs = evaluate_test(model,test_text,test_labels,pad_len=pad_len,mode='test',return_predictions=False)
  return get_top_k_accuracy(y_probs,test_labels)