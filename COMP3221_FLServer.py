from _thread import *
import threading
import time
import datetime
import socket
import _thread
import json 
import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt

import pickle
import numpy as np

IP = "127.0.0.1"
START = False
FIRST = False
GLOBAL_ITERATIONS = 100
END = False
LOCK = threading.Lock()
new_local_models = []


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
        self.client_list = {}       # key is id, value is data_size, port_no, is_added_after_init
        self.model = MCLR() # initialize global model

        self.average_loss = -1
        self.average_accuracy = -1


    def aggregate_parameters(self):
        global new_local_models

        print("Aggregating new global model")
        # Clear global model before aggregation
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
            
        total_train_samples = 0
        # sum of all client data size
        for client in self.client_list:
            total_train_samples += self.client_list[client][0]


        # randomly chosen models
        chosen_models = []
        if self.sub_client:
            # aggregate two random models from clients
            chosen_models = random.sample(new_local_models, 2)
        else:
            chosen_models = new_local_models

        # aggregate the chosen models
        for data in chosen_models:
            # data is a matrix
            client = data[0][0]
            client_model = data[1]
            client_train_samples = self.client_list[client][0]

            # print(type(client_train_samples), type(total_train_samples))

            for server_param, user_param in zip(self.model.parameters(), client_model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * client_train_samples / total_train_samples

        new_local_models = [] # reset client local models for next global round


    def evaluate(self):
        total_accuracy = 0
        total_loss = 0
        for model in new_local_models:
            loss = float(model[0][1])
            accuracy = float(model[0][2])
            total_accuracy += accuracy
            total_loss += loss
        return total_accuracy/len(new_local_models), total_loss/len(new_local_models)


    def handle_client(self):
        global FIRST, new_local_models, END
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # set timeout to stop this thread
                s.settimeout(5) # this is for when the global iterations are ended
                s.bind((IP, self.port_no)) # Bind to the port
                s.listen(5)
                while True:
                    try:
                        c, addr = s.accept()
                        data_rev = c.recv(1024)

                        if len(data_rev) == 0 or not data_rev:
                            print("Message is empty")

                        else:
                            # determine message type
                            # hs is a handshake message
                            # otherwise it's a model message

                            client_info = pickle.loads(data_rev).split(" ")
                            

                            message_type = client_info[0]
                            if message_type == "hs":
                                # print("Received handshake message from client: ", client_info[1])
                                # print(client_info)
                                data_size, client_id, port_no = client_info[1:]

                                # register client to client_list
                                # critical section
                                # LOCK.acquire()
                                if START:
                                    self.client_list[client_id] = (int(data_size), int(port_no), 0)
                                else:
                                    self.client_list[client_id] = (int(data_size), int(port_no), 1)

                            elif message_type == "sm":
                                client_id = client_info[1]
                                print("Getting local model from client", client_id[-1:])
                                # print(client_id, len(client_id))
                                client_model_data = b''
                                while True:
                                    client_model_part_data = c.recv(4096)   
                                    client_model_data += client_model_part_data
                                    if len(client_model_part_data) == 0 or not client_model_part_data:
                                        # print("No more messages")
                                        break

                                client_model_decode = pickle.loads(client_model_data)
                                new_local_model = [client_info[1:], client_model_decode]
                                new_local_models.append(new_local_model)
                        

                            if not FIRST:
                                print("First client connected\n")
                                # waits for 30 seconds after receiving first handshake_message
                                FIRST = True
                        c.close()
                    except socket.timeout:
                        if END: break
                    
                s.close()
                
        except Exception as e:
            print("Server Socket issue or Server Socket closed")
            print(e)



    def send_parameters(self, global_model_data, average_training_data):
        # print("Sending model to all registered clients")

        # broadcast global model to all clients
        client_list = copy.deepcopy(self.client_list)
        for client in client_list:
            if client_list[client][2] == 0:
                # wait for next global round
                new_value = (client_list[client][0], client_list[client][1], 1)
                self.client_list[client] = new_value


            client_port_no = client_list[client][1]
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s: # Create a socket object
                    s.connect((IP, client_port_no))             
                    s.sendall(average_training_data)

                    time.sleep(1)
                    s.sendall(global_model_data)
                    # print("Sent global model to client: ", client)
                    # print("Sent global model")

                    time.sleep(1)   # delay before closing socket
                    s.close()
            except Exception as e:
                # client might be not alive
                print(f'Client {client} not active: + {e}')


    def plot_figure(self, loss, acc):
        
        figure, axis = plt.subplots(2, 1)
        axis[0].plot(loss, label="FedAvg")
        axis[0].set_xlabel("Global rounds")
        axis[0].set_ylabel("Loss")
        axis[0].legend(loc="upper right")

        axis[1].plot(acc, label="FedAvg")
        # set y limit
        axis[1].set_ylim(20, 100)
        # set y tick to 10, from 10 to 100
        axis[1].set_yticks(np.arange(10, 100, 10))
        
        axis[1].set_xlabel("Global rounds")
        axis[1].set_ylabel("Accuracy")
        axis[1].legend(loc='lower right')


        plt.show()


    def fedAvg(self):
        global GLOBAL_ITERATIONS, START, END

        loss = []
        acc = []
        while True:
            if START:
                print("Start to run fedAvg\n")
                # Runing FedAvg
                
                # send global model to client initially
                average_train_data = pickle.dumps(str(self.average_loss) + " " + str(self.average_accuracy))
                global_model_data = pickle.dumps(self.model)
                self.send_parameters(global_model_data, average_train_data)

                for glob_iter in range(GLOBAL_ITERATIONS):
                    print(f"Global Iteration {glob_iter + 1}:")
                    print("Total Number of clients:", len(self.client_list))

                    # wait until all models are received from clients
                    while True:
                        # current round active clients
                        active_clients = 0
                        clients = copy.deepcopy(self.client_list)
                        for client in clients:
                            if clients[client][2] == 1:
                                active_clients += 1
                        if len(new_local_models) == active_clients:
                            break
                    
                    # print("Received", len(new_local_models), "new local models")

                    # Evaluate the global model across all clients
                    avg_acc, avg_loss = self.evaluate()
                    self.average_loss = avg_loss
                    self.average_accuracy = avg_acc
                    acc.append(avg_acc)
                    loss.append(avg_loss)
                    # print("Global Round:", glob_iter + 1, "Average accuracy across all clients : ", avg_acc)
                    
                    # Aggregate all clients model to obtain new global model 
                    self.aggregate_parameters()
                    
                    print("Broadcasting new global model")
                    # send global model to all clients in the client_list
                    average_train_data = pickle.dumps(str(self.average_loss) + " " + str(self.average_accuracy))
                    global_model_data = pickle.dumps(self.model)
                    # print(len(global_model_data))
                    
                    if glob_iter == GLOBAL_ITERATIONS - 1:
                        # send stop message as well
                        average_train_data = pickle.dumps(str(self.average_loss) + " " + str(self.average_accuracy) + " " + "stop")

                    self.send_parameters(global_model_data, average_train_data)
                    print("\n")

                print()
                print("Completed all global iterations")
                print("Average accuracy:", sum(acc)/len(acc))
                print("Average loss:", sum(loss)/len(loss))

                self.plot_figure(loss, acc)

                END = True
                break
    
    def wait_for_30_seconds(self):
        global FIRST, START
        while (1):
            if FIRST:
                time.sleep(10)
                # print("30 SECONDS PASSED")
                START = True
                break
    

    def run(self):

        threading.Thread(target = self.wait_for_30_seconds, args=()).start()
        threading.Thread(target=self.handle_client, args=()).start()
        self.fedAvg()
        
        

if "__main__" == __name__:
    if len(sys.argv) != 3:
        print("Usage: python3 COMP3221_FLServer.py <Port-Server> <Sub-client>")
        exit(1)
    port_no, sub_client = map(int, sys.argv[1:])

    server = FLServer(port_no, sub_client == 1)
    # print(server.port_no, server.sub_client)
    server.run()