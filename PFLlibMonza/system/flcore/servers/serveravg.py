import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np
from collections import Counter
import torch
import csv
import os
class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.fpr_frr_results = []

        # Open the CSV file in append mode to save results over time
        if self.cc ==3:
            self.csv_filename = 'fpr_frr_results_3.csv'
        if self.cc ==2:
            self.csv_filename = 'fpr_frr_results_2.csv'
        else:
            self.csv_filename = 'f.csv'
        # Write headers if the file is empty (first time writing)
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Round', 'FPR', 'FRR'])
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        
    def save_fpr_frr_to_csv(self, round_number, FPR, FRR):
        """
        Saves the FPR and FRR results to a CSV file for each round.
        """
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([round_number, FPR, FRR])

    def normalize_entropies(self, client_entropies):
        """Normaliza as entropias para que fiquem no intervalo [0, 1]"""
        # Obter as entropias
        entropies = np.array(list(client_entropies.values()))

        # Calcular o valor mínimo e máximo
        min_entropy = np.min(entropies)
        max_entropy = np.max(entropies)

        # Normalizar as entropias
        normalized_entropies = (entropies - min_entropy) / (max_entropy - min_entropy)

        # Atualizar o dicionário com as entropias normalizadas
        normalized_client_entropies = {client_id: normalized_entropy for client_id, normalized_entropy in zip(client_entropies.keys(), normalized_entropies)}

        # Exibir as entropias normalizadas
        for client_id, normalized_entropy in normalized_client_entropies.items():
            print(f"Normalized Shannon entropy for client {client_id}: {normalized_entropy:.4f}")

        return normalized_client_entropies
    def set_client_quarantine(self, client_id):
        self.client_quarantine_dict[client_id]['quarentena'] = self.client_quarantine_dict[client_id]['quarentena'] +1
        self.client_quarantine_dict[client_id]['roundsQuarent'] = self.client_quarantine_dict[client_id]['quarentena'] *2

    def decrease_quarentine(self, client_id):
        if self.client_quarantine_dict[client_id]['roundsQuarent'] ==0:
            self.client_quarantine_dict[client_id]['roundsQuarent'] = 0
        else:
            self.client_quarantine_dict[client_id]['roundsQuarent'] = self.client_quarantine_dict[client_id]['roundsQuarent'] -1
    def compute_fpr_frr(self):
        """
        Calcula False Positive Rate (FPR) e False Rejection Rate (FRR)
        usando self.client_quarantine_dict e self.index_malicious.
        """
        FP = 0  # Falsos positivos: clientes em quarentena mas não maliciosos
        TP = 0  # Verdadeiros positivos: clientes em quarentena e maliciosos
        FN = 0  # Falsos negativos: maliciosos não detectados
        TN = 0  # Verdadeiros negativos: não maliciosos e não em quarentena

        for client_id in range(self.num_clients):
            in_quarantine = self.client_quarantine_dict[client_id]['roundsQuarent'] > 0
            is_malicious = client_id in self.index_malicious

            if in_quarantine and not is_malicious:
                FP += 1
            elif in_quarantine and is_malicious:
                TP += 1
            elif not in_quarantine and is_malicious:
                FN += 1
            elif not in_quarantine and not is_malicious:
                TN += 1

        # Evitar divisão por zero
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FRR = FN / (FN + TP) if (FN + TP) > 0 else 0

        return FPR, FRR

    def compute_fpr_frr_cluster(self, removed_clients, cluster_tuples):
        """
        Calcula FPR e FRR com base nos clientes removidos do cluster.
        """
        FP = 0  # Falsos positivos: clientes não maliciosos removidos
        TP = 0  # Verdadeiros positivos: maliciosos removidos
        FN = 0  # Falsos negativos: maliciosos não removidos
        TN = 0  # Verdadeiros negativos: não maliciosos não removidos

        # Comparar os clientes removidos com a lista de maliciosos
        for client_id in removed_clients:
            is_malicious = client_id in self.index_malicious  # Verificar se é malicioso
            if is_malicious:
                TP += 1  # Cliente malicioso corretamente removido
            else:
                FP += 1  # Cliente não malicioso removido erroneamente

        # Verificar os clientes que não foram removidos (ainda estão no cluster)
        for client_id, cluster in cluster_tuples:
            if client_id not in removed_clients:
                is_malicious = client_id in self.index_malicious
                if is_malicious:
                    FN += 1  # Cliente malicioso não removido
                else:
                    TN += 1  # Cliente não malicioso não removido

        # Calcular FPR e FRR
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FRR = FN / (FN + TP) if (FN + TP) > 0 else 0

        return FPR, FRR

    def train(self):
        
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            self.removed_clients = []
            self.cluster_tuples = ()
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
            for j in range(self.num_clients):
                self.decrease_quarentine(j)

            #for client in self.selected_clients:
            #    client.train()

            threads = [Thread(target=client.train)
                       for client in self.selected_clients]
            [t.start() for t in threads]
            [t.join() for t in threads]

            self.receive_models()
            if i>0:
                #comparar com o modelo
                if self.cc==0:
                    global_model_params = list(self.global_model.parameters()) 
                # Calcular a similaridade de cosseno entre os modelos dos clientes e o modelo global
                    similarities = self.calculate_similarity_with_global_model(global_model_params)
                    for sim in similarities:
                        print(f"Cosine similarity between client {sim[0]} and the global model: {sim[1]:.4f}")
                #comparar com todos os modelos, esse não funciona no momento
                if self.cc==1:
                    similarity_scores = self.calculate_similarity_scores()
                    for client_id, score in similarity_scores.items():
                        print(f"Cosine similarity for client {client_id}: {score:.4f}")
                    normalized_client_entropies = self.normalize_entropies(similarity_scores)
                #comparar com todos os modelos e fazer cluster
                if self.cc==2:
                    oi = time.time()
                    similarity_matrix, a = self.calculate_similarity_scores()

                    # Realizar a clusterização
                    num_clusters = 2  # Defina o número de clusters conforme necessário
                    clusters = self.perform_clustering(similarity_matrix, num_clusters)
                    #for idx, cluster in enumerate(clusters):
                        #print(f"Client {self.ids[idx]} is in cluster {cluster}")

                    self.cluster_tuples = [(self.ids[idx], cluster) for idx, cluster in enumerate(clusters)]
                    for idx, cluster in enumerate(clusters):
                        print(f"Client {self.ids[idx]} is in cluster {cluster}")
                    cluster_counts = Counter([cluster for _, cluster in self.cluster_tuples])
                    min_cluster = min(cluster_counts, key=cluster_counts.get)

                    for idx in range(len(self.cluster_tuples) - 1, -1, -1):
                        client_id, cluster = self.cluster_tuples[idx]
                        #print(self.ids)
                        if cluster == min_cluster:
                            print(f"Removing client {client_id} from cluster {cluster}")
                            self.removed_clients.append(client_id)
                            # Remover o cliente das listas associadas
                            del self.uploaded_models[idx]
                            del self.ids[idx]
                            del self.uploaded_ids[idx]
                            del self.uploaded_weights[idx]
                            #print(self.ids)
                    self.uploaded_weights = [weight / sum(self.uploaded_weights) for weight in self.uploaded_weights]
                    bye = time.time()
                    vish = bye- oi  # Calcula o tempo decorrido
                    print(f"Tempo de execução: {vish:.4f} segundos")
                #metodo do cosseno mas com score
                if self.cc==3:
                    oi = time.time()
                    similarity_matrix, client_scores  = self.calculate_similarity_scores()
                    # Converte os scores para array e calcula a média
                    scores_array = np.array(list(client_scores.values()))
                    mean_score = np.mean(scores_array)
                    std_score = np.std(scores_array)
                    print(f"Average score: {mean_score:.4f}")
                    mean_score = mean_score - std_score
                    print(f"Average score: {mean_score:.4f}")
                    # Cria uma lista de tuplas para manter a posição dos clientes
                    client_tuples = [(self.ids[idx], client_scores[self.ids[idx]]) for idx in range(len(self.ids))]
                    total = len(self.index_malicious)
                    a = 0
                    # Itera de trás para frente para remover clientes abaixo da média
                    if std_score<0.001:
                        print("nenhum malicioso")
                    else:
                        for idx in range(len(client_tuples) - 1, -1, -1):
                            client_id, score = client_tuples[idx]
                            print(f"Esse  {client_id} with score {score:.4f} ")
                            if score < mean_score:
                                if client_id in self.index_malicious:
                                    a = a+1
                                print(f"Removing client {client_id} with score {score:.4f} (below average)")
                                self.set_client_quarantine(client_id)
                                # Remover o cliente das listas associadas
                                del self.uploaded_models[idx]
                                del self.ids[idx]
                                del self.uploaded_ids[idx]
                                del self.uploaded_weights[idx]
                    a = (a/total) *100
                    print("porcentagem de clientes maliciosos de verdade achados: "+ str(a) + "%")
                    self.uploaded_weights = [weight / sum(self.uploaded_weights) for weight in self.uploaded_weights]
                    bye = time.time()
                    vish = bye - oi  # Calcula o tempo decorrido
                    print(f"Tempo de execução: {vish:.4f} segundos")
                
                if self.cc ==4:
                    oi = time.time()
                    k = 3
                    client_entropies = self.calculate_client_entropies()
                    entropies = np.array(list(client_entropies.values()))
                    mean_entropy = np.mean(entropies)
                    std_entropy = np.std(entropies)
                    lower_bound = mean_entropy - std_entropy
                    upper_bound = mean_entropy + std_entropy-(std_entropy/2)
                    
                    print(f"Mean entropy: {mean_entropy:.4f}, Std: {std_entropy:.4f}")
                    print(f"Keeping clients with entropy in [{lower_bound:.4f}, {upper_bound:.4f}]")
                    
                    # 3. Lista de tuplas para manter índice
                    client_tuples = [(self.ids[idx], client_entropies[self.ids[idx]]) for idx in range(len(self.ids))]

                    # 4. Remover outliers (de trás para frente)
                    for idx in range(len(client_tuples) - 1, -1, -1):
                        client_id, entropy = client_tuples[idx]
                        if entropy < lower_bound or entropy > upper_bound:
                            print(f"Removing client {client_id} with entropy {entropy:.4f} (outlier)")

                            # Remover das listas associadas
                            del self.uploaded_models[idx]
                            del self.ids[idx]
                            del self.uploaded_ids[idx]
                            del self.uploaded_weights[idx]
                    #normalized_client_entropies = self.normalize_entropies(client_entropies)
                    bye = time.time()
                    vish = bye - oi  # Calcula o tempo decorrido
                    print(f"Tempo de execução: {vish:.4f} segundos")
                if self.cc==5:
                    print("vai rolar nada")
            print(self.client_quarantine_dict)
            FPR=0
            FRR = 0
            if self.cc ==2:
                FPR, FRR = self.compute_fpr_frr_cluster(self.removed_clients, self.cluster_tuples)
            if self.cc ==3:
                FPR, FRR = self.compute_fpr_frr()
            print(f"Round {i}: False Positive Rate = {FPR:.4f}, False Rejection Rate = {FRR:.4f}")
            self.save_fpr_frr_to_csv(i, FPR, FRR)
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
