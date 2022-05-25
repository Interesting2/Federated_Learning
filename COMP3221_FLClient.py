from _thread import *
import threading
import time
import socket
import _thread
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt

import pickle
from COMP3221_FLServer import MCLR

IP = "127.0.0.1"
PORT = 6000
RECEIVED = False
EPOCHS = 2

class FLClient():
    def __init__(self, client_id, port_no, opt_method):
        self.id = client_id
        self.port_no = port_no
        self.opt_method = opt_method    # 0 is GD, 1 is Mini-batch-GD

        self.X_train, self.y_train, self.X_test, self.y_test, self.train_samples, self.test_samples = self.load_data()
        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.test_data = [(x, y) for x, y in zip(self.X_test, self.y_test)]

        self.trainloader = None
        self.testloader = DataLoader(self.test_data, self.test_samples)

        if (self.opt_method == 1):
            self.trainloader = DataLoader(self.train_data, 5)   # mini-batch size is 5
        else:
            self.trainloader = DataLoader(self.train_data, self.train_samples)

        self.learning_rate = 0.01
        self.loss = nn.NLLLoss()

        self.model = None
        self.optimizer = None
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate) 
        print("Client: Done initializing")


    # receive global model from server
    def set_parameters(self, model):
        if self.model is None: self.model = copy.deepcopy(model)
        else:
            for old_param, new_param in zip(self.model.parameters(), model.parameters()):
                old_param.data = new_param.data

    # train to obtain new local model  
    def train(self, epochs):
        LOSS = 0
        self.model.train()
        for epoch in range(1, epochs + 1):
            self.model.train()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        return loss.data
    

    # test the global model received from server
    def test(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloader:
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y) * 1. / y.shape[0]).item()
            print(str(self.id) + ", Accuracy of",self.id, " is: ", test_acc)
        return test_acc


    def load_data(self):
        train_path = os.path.join("FLdata", "train", "mnist_train_" + str(self.id) + ".json")
        test_path = os.path.join("FLdata", "test", "mnist_test_" + str(self.id) + ".json")
        train_data = {}
        test_data = {}

        with open(os.path.join(train_path), "r") as f_train:
            train = json.load(f_train)
            train_data.update(train['user_data'])
        with open(os.path.join(test_path), "r") as f_test:
            test = json.load(f_test)
            test_data.update(test['user_data'])

        X_train, y_train, X_test, y_test = train_data['0']['x'], train_data['0']['y'], test_data['0']['x'], test_data['0']['y']
        X_train = torch.Tensor(X_train).view(-1, 1, 28, 28).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, 1, 28, 28).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
        train_samples, test_samples = len(y_train), len(y_test)
        return X_train, y_train, X_test, y_test, train_samples, test_samples


    def send_handshake(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: # Create a socket object
                s.connect((IP, PORT))
                # Create handshake message that contains data size and id
                mess_data = bytes(str(self.train_samples) + " " + str(self.id) + " " + str(self.port_no), encoding= 'utf-8')
                s.sendall(mess_data)

                time.sleep(1)   # delay before closing the socket
                s.close()    
        except Exception as e:
            print("Client handshake error: ", e)


    def send_model(self):
        global RECEIVED
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: # Create a socket object
                s.connect((IP, PORT))
                while(1):
                    # create message containing id and model
                    if RECEIVED:  
                        print("Ready to send local model to server")                  
                        RECEIVED = False
                        mess_data = bytes(str(self.id) + " " + str(self.model), encoding= 'utf-8')
                        s.sendall(mess_data)
                s.close()
        except Exception as e:
            print("Client send model error: ", e)


    def receive_model(self):
        global RECEIVED, EPOCHS
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((IP, self.port_no)) # Bind to the port
                s.listen(1)
                while True:
                    c, addr = s.accept()
                    
                    print("Receiving model from server")
                    global_model_data = b''
                    while True:
                        global_model_part_data = c.recv(4096)
                        global_model_data += global_model_part_data
                        if len(global_model_part_data) == 0 or not global_model_part_data:
                            print("No more messages")
                            break

                    global_model_decode = pickle.loads(global_model_data)
                    print(len(global_model_data))
                    print("Global model received from server: ", global_model_decode)

                    if not global_model_data:
                        print("didn't get data")
                        break
                    
                    # set global model
                    self.set_parameters(global_model_decode)

                    # eval local test data with global model
                    # logs the training loss and accuracy of the global model
                    accuracy = self.test()

                    # # train the model
                    loss = self.train(EPOCHS)

                    print("Training loss: ", loss)
                    print("Testing Accuracy: ", accuracy)

                    # RECEIVED = True
                s.close()
                
        except Exception as e:
            print("Client receive model from server issue: " + str(e))
        


    def run(self):
        self.send_handshake()

        t1 = threading.Thread(target=self.receive_model, args=())
        t2 = threading.Thread(target=self.send_model, args=())

        t1.start()
        t2.start()

        t1.join()
        t2.join()
        while True:
            print("Client waiting ")
            time.sleep(10)

        return

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 COMP3221_FLClient.py <Client-id> <Port-Client> <Opt-Method>")
        exit(1)
    
    client_id = sys.argv[1]
    port_no, opt_method = map(int, sys.argv[2:])
    client = FLClient(client_id, port_no, opt_method)
    client.run()
