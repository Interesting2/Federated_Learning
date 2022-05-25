from _thread import *
import threading
import time
import datetime
import socket
import _thread
import json 
import sys
import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt

import pickle

IP = "127.0.0.1"
START = False
FIRST = False
GLOBAL_ITERATIONS = 100


# MultiClass Logistic Regression
class MCLR(nn.Module):
    def __init__(self):
        super(MCLR, self).__init__()
        # Create a linear transformation to the incoming data
        # Input dimension: 784 (28 x 28), Output dimension: 10 (10 classes)
        self.fc1 = nn.Linear(784, 10)


    # Define how the model is going to be run, from input to output
    def forward(self, x):
        # Flattens input by reshaping it into a one-dimensional tensor. 
        x = torch.flatten(x, 1)
        # Apply linear transformation
        x = self.fc1(x)
        # Apply a softmax followed by a logarithm
        output = F.log_softmax(x, dim=1)
        return output


# Federated Learning Server
class FLServer():
    def __init__(self, *args):
        self.port_no, self.sub_client = args
        self.client_list = {}       # key is id, value is data_size
        self.model = MCLR() # initialize global model

        # print(self.model)


    def aggregate_parameters(self, server_model, users, total_train_samples):
        # Clear global model before aggregation
        for param in server_model.parameters():
            param.data = torch.zeros_like(param.data)
            
        for user in users:
            for server_param, user_param in zip(server_model.parameters(), user.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * user.train_samples / total_train_samples
        return server_model

    def evaluate(self, user):
        total_accurancy = 0
        for user in users:
            total_accurancy += user.test()
        return total_accurancy/len(users)

    def fedAvg(self):
        global GLOBAL_ITERATIONS
        # Runing FedAvg
        loss = []
        acc = []

        for glob_iter in range(GLOBAL_ITERATIONS):
            # TODO: Broadcast global model to all clients
            self.send_parameters(server_model, users)
            
            # Evaluate the global model across all clients
            avg_acc = self.evaluate(users)
            acc.append(avg_acc)
            print("Global Round:", glob_iter + 1, "Average accuracy across all clients : ", avg_acc)
            
            # Each client keeps training process to  obtain new local model from the global model 
            avgLoss = 0
            for user in users:
                avgLoss += user.train(1)
            # Above process training all clients and all client paricipate to server, how can we just select subset of user for aggregation
            loss.append(avgLoss)
            
            # TODO:  Aggregate all clients model to obtain new global model 
            self.aggregate_parameters(server_model, users, total_train_samples)


    def handshake_handler(self, c):
        global FIRST

        try:
            data_rev = c.recv(1024)
            if len(data_rev) == 0 or not data_rev:
                print("No Handshake message received")
                c.close()
                return

            client_info = data_rev.decode('utf-8')
            print("Decoded Handshake message: ", client_info)
            data_size, client_id, port_no = client_info.split(" ")

            # register client to client_list
            # key is client_id, and value is tuple of data_size and port_no
            self.client_list[client_id] = (int(data_size), int(port_no))

            if not FIRST:
                print("First client connected")
                # waits for 30 seconds after receiving first handshake_message
                FIRST = True

        except socket.error as se:
            print("Socket error")
        except Exception as ee:
            print("serverHandler error | Handshake Error: " + str(ee))


    def send_parameters(self, global_model_data):
        print("Sending model to all registered clients")

        # broadcast global model to all clients
        for client in self.client_list:
            client_port_no = self.client_list[client][1]
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: # Create a socket object
                    s.connect((IP, client_port_no))             
                    s.sendall(global_model_data)
                    print("Sent global model to client: ", client)

                    time.sleep(1)   # delay before closing socket
                    s.close()
            except Exception as e:
                print("Client send model error: ", e)

            print("Sent global model")

        # for user in users:
            # user.set_parameters(server_model)


    # handling new client
    def serverHandler(self, c , addr):
        global START
        
        self.handshake_handler(c)

        while(True): 
            if START:
                print("Start server main process")

                # # send global model to all clients in the client_list

                global_model_data = pickle.dumps(self.model)
                print(len(global_model_data))
                # print("Global model encoded: ", global_model_data)
                self.send_parameters(global_model_data)
                break


                # # receive client local model
                # clientLocalModel = c.recv(1024)
                # if len(data_rev) == 0 or not data_rev:
                #     print("No local model received from client")
                #     break

                # clientLocalModelRecv = data_rev.decode('utf-8')

                
        c.close()
    
    def wait_for_30_seconds(self):
        global FIRST, START
        while (1):
            if FIRST:
                time.sleep(5)
                print("30 SECONDS PASSED")
                START = True
                break

    def run(self):

        threading.Thread(target = self.wait_for_30_seconds, args=()).start()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((IP, self.port_no)) # Bind to the port
                s.listen(5)
                while True:
                    c, addr = s.accept()
                    threading.Thread(target=self.serverHandler, args=(c, addr)).start()

                s.close()
                
        except Exception as e:
            print("Server Socket issue or Server Socket closed")
        
        return

if "__main__" == __name__:
    if len(sys.argv) != 3:
        print("Usage: python3 COMP3221_FLServer.py <Port-Server> <Sub-client>")
        exit(1)
    port_no, sub_client = map(int, sys.argv[1:])

    server = FLServer(port_no, sub_client == 1)
    # print(server.port_no, server.sub_client)
    server.run()