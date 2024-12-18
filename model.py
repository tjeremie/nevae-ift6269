import torch
import torch_geometric
import random

NUM_TYPES = 4
NUM_GCN_LAYERS = 3
NEG_EDGES = 30

VALENCIES = {0:4,1:1,2:6,3:5}

def get_capacity(node_types,edge_list,edge_weights) :
    degs={i:VALENCIES[node_types[i].item()] for i in range(len(node_types))}
    for j in range(len(edge_list)) :
        degs[edge_list[j][0]]-=int(edge_weights[j].item())
        degs[edge_list[j][1]]-=int(edge_weights[j].item())
    
    return degs

class Model(torch.nn.Module) :
    def __init__(self,hidden_size) :
        super(Model, self).__init__()

        self.hidden_size=hidden_size

        self.lambda_n=torch.nn.Parameter(torch.tensor(21.), requires_grad=True)
        self.GCNlayers=torch.nn.ModuleList([torch_geometric.nn.GCNConv(3+NUM_TYPES,3+NUM_TYPES) for i in range(NUM_GCN_LAYERS)])
        
        self.linear_mu_1=torch.nn.Linear((3+NUM_TYPES)*(NUM_GCN_LAYERS+1),(3+NUM_TYPES)*(NUM_GCN_LAYERS+1))
        self.linear_mu_2=torch.nn.Linear((3+NUM_TYPES)*(NUM_GCN_LAYERS+1),self.hidden_size)

        self.linear_sigma_1=torch.nn.Linear((3+NUM_TYPES)*(NUM_GCN_LAYERS+1),(3+NUM_TYPES)*(NUM_GCN_LAYERS+1))
        self.linear_sigma_2=torch.nn.Linear((3+NUM_TYPES)*(NUM_GCN_LAYERS+1),self.hidden_size)

        self.linear_t_1=torch.nn.Linear(self.hidden_size+NUM_TYPES,self.hidden_size+NUM_TYPES)
        self.linear_t_2=torch.nn.Linear(self.hidden_size+NUM_TYPES,1)

        self.rnn_l=torch.nn.RNN(self.hidden_size,self.hidden_size)
        self.linear_l=torch.nn.Linear(self.hidden_size,1)

        self.linear_e_1=torch.nn.Linear(2*self.hidden_size,2*self.hidden_size)
        self.linear_e_2=torch.nn.Linear(2*self.hidden_size,1)

        self.linear_r_1=torch.nn.Linear(2*self.hidden_size,2*self.hidden_size)
        self.linear_r_2=torch.nn.Linear(2*self.hidden_size,3)

        self.GCN_s=torch_geometric.nn.GCNConv(self.hidden_size,self.hidden_size)

        self.linear_s_mu_1=torch.nn.Linear(self.hidden_size,self.hidden_size)
        self.linear_s_mu_2=torch.nn.Linear(self.hidden_size,3)

        self.linear_s_sigma_1=torch.nn.Linear(self.hidden_size,self.hidden_size)
        self.linear_s_sigma_2=torch.nn.Linear(self.hidden_size,3)
        
    def loss(self, graph) :
        num_nodes=len(graph["t"])
        edges=graph["e"]

        random.shuffle(edges)

        edge_list=[(a,b) for a,b,_ in edges]

        edge_list_doubled=edge_list+[(b,a) for a,b,_ in edges]
        edge_index=torch.tensor(edge_list_doubled,dtype=torch.long).t().contiguous()
        edge_weights=torch.tensor([w for _,_,w in edges]+[w for _,_,w in edges])


        # Loss for n
        m_ll_n=torch.nn.PoissonNLLLoss(log_input=False)(self.lambda_n,torch.tensor(num_nodes))

        # GNN encoder
        gcn_embeds=[torch.cat((graph["s"],torch.nn.functional.one_hot(graph["t"],NUM_TYPES)),dim=1)]
        for i in range(NUM_GCN_LAYERS) :
            gcn_embeds.append(torch.nn.ReLU()(self.GCNlayers[i](gcn_embeds[-1],edge_index,edge_weight=edge_weights)))

        enc_mu=self.linear_mu_2(torch.nn.ReLU()(self.linear_mu_1(torch.cat(gcn_embeds,dim=1))))
        enc_sigma=torch.nn.ReLU()(self.linear_sigma_2(torch.nn.ReLU()(self.linear_sigma_1(torch.cat(gcn_embeds,dim=1)))))
        
        # Sample Z using reparametrization trick
        eps=torch.normal(mean=torch.zeros((num_nodes,self.hidden_size)),std=torch.ones((num_nodes,self.hidden_size)))
        z=enc_mu+eps*enc_sigma

        m_ll_q=torch.nn.GaussianNLLLoss(reduction="sum")(z,enc_mu,enc_sigma)

        m_ll_z=torch.nn.GaussianNLLLoss(reduction="sum")(z,torch.zeros((num_nodes,self.hidden_size)),torch.ones((num_nodes,self.hidden_size)))

        # Loss for t
        z_and_onehot_t=torch.cat((z.unsqueeze(1).expand(num_nodes,NUM_TYPES,self.hidden_size),torch.eye(NUM_TYPES).unsqueeze(0).repeat(num_nodes,1,1)),dim=2)
        t_logs=self.linear_t_2(torch.nn.ReLU()(self.linear_t_1(z_and_onehot_t)))
        softmax_t=torch.nn.LogSoftmax(dim=1)(t_logs)
        m_ll_t=-softmax_t[torch.arange(num_nodes).unsqueeze(1),graph["t"].unsqueeze(1)].sum()

        # Loss for l
        num_edges=len(edges)
        m_ll_l=torch.nn.PoissonNLLLoss(log_input=True)(self.linear_l(self.rnn_l(z)[1].squeeze()).squeeze(),torch.tensor(num_edges))

        # Loss for e
        all_neg_edges=list(set((a,b) for a in range(num_nodes) for b in range(a+1,num_nodes))-set(edge_list))
        random.shuffle(all_neg_edges)
        neg_edges=all_neg_edges[:NEG_EDGES]
        neg_edges_count=len(neg_edges)
        
        first_coords=[a for a,_ in edge_list+neg_edges]
        second_coords=[b for _,b in edge_list+neg_edges]
        zpairs=torch.concat((z[first_coords],z[second_coords]),dim=1)

        edge_logs=self.linear_e_2(torch.nn.ReLU()(self.linear_e_1(zpairs))).squeeze()
        
        reweighting_factor=num_nodes*(num_nodes-1)/2-num_edges+(num_nodes*(num_nodes-1)/2-num_edges)/neg_edges_count
        reweighting=torch.log(torch.tensor([1 if j<num_edges else reweighting_factor for j in range(num_edges+neg_edges_count)]))

        m_ll_e=torch.tensor(0., requires_grad = True)
        for i in range(num_edges) :
            capacities=get_capacity(graph["t"],edge_list[:i],edge_weights)
            edge_mask=[min(capacities[a],capacities[b])>0 for a,b in edge_list[i:]+neg_edges]
            if not edge_mask[0] :
                raise Exception("Illegal edge")
            m_ll_e=m_ll_e-torch.nn.LogSoftmax(dim=0)((edge_logs[i:]+reweighting[i:])[edge_mask])[0]

        # Loss for r
        zpairs_r=torch.concat((z[first_coords[:num_edges]],z[second_coords[:num_edges]]),dim=1)
        r_logs=self.linear_r_2(torch.nn.ReLU()(self.linear_r_1(zpairs_r)))
        
        m_ll_r=torch.tensor(0., requires_grad = True)
        for i in range(num_edges) :
            capacities=get_capacity(graph["t"],edge_list[:i],edge_weights)
            weight_mask=(torch.tensor([1,2,3])<=min(capacities[edge_list[i][0]],capacities[edge_list[i][1]]))
            m_ll_r=m_ll_r-torch.nn.LogSoftmax(dim=0)(r_logs[i][weight_mask])[int(edge_weights[i].item())-1]
        
        # Loss for s
        agg_z=torch.nn.ReLU()(self.GCN_s(z,edge_index,edge_weight=edge_weights))

        s_mu=self.linear_s_mu_2(torch.nn.ReLU()(self.linear_s_mu_1(agg_z)))
        s_sigma_vect=self.linear_s_sigma_2(torch.nn.ReLU()(self.linear_s_sigma_1(agg_z)))

        m_ll_s=torch.tensor(0., requires_grad = True)
        for i in range(num_nodes) :
            mu=s_mu[i]
            sigma_vect=s_sigma_vect[i].unsqueeze(0)
            sigma=torch.matmul(sigma_vect.T,sigma_vect)+torch.eye(3)*0.0001
            m_ll_s=m_ll_s-torch.distributions.multivariate_normal.MultivariateNormal(mu,sigma).log_prob(graph["s"][i])

        return m_ll_n+m_ll_z+m_ll_t+m_ll_l+m_ll_e+m_ll_r+m_ll_s-m_ll_q
    
    def sample(self) :
        # Sample n
        num_nodes=int(torch.poisson(self.lambda_n).item())

        # Sample z
        z=torch.normal(mean=torch.zeros((num_nodes,self.hidden_size)),std=torch.ones((num_nodes,self.hidden_size)))

        # Sample t
        z_and_onehot_t=torch.cat((z.unsqueeze(1).expand(num_nodes,NUM_TYPES,self.hidden_size),torch.eye(NUM_TYPES).unsqueeze(0).repeat(num_nodes,1,1)),dim=2)
        t_logs=self.linear_t_2(torch.nn.ReLU()(self.linear_t_1(z_and_onehot_t)))
        prob_t=torch.exp(torch.nn.LogSoftmax(dim=1)(t_logs))
        t=torch.multinomial(prob_t.squeeze(2),1).squeeze(1)

        # Sample l
        num_edges=max(min(int(torch.poisson(torch.exp(self.linear_l(self.rnn_l(z)[1].squeeze()).squeeze())).item()),num_nodes*(num_nodes-1)/2),1)

        # Sample e and r
        all_possible_edges=list(set((a,b) for a in range(num_nodes) for b in range(a+1,num_nodes)))
        
        first_coords=[a for a,_ in all_possible_edges]
        second_coords=[b for _,b in all_possible_edges]
        zpairs=torch.concat((z[first_coords],z[second_coords]),dim=1)

        edge_logs=self.linear_e_2(torch.nn.ReLU()(self.linear_e_1(zpairs))).squeeze()

        edge_list=[]
        edge_weights=torch.empty(0)
        for i in range(num_edges) :
            capacities=get_capacity(t,edge_list,edge_weights)
            available_edges=[edge for edge in all_possible_edges if ((edge not in edge_list) and (min(capacities[edge[0]],capacities[edge[1]])>0))]
            available_indexes=[all_possible_edges.index(edge) for edge in available_edges]
            
            edge_probs=torch.exp(torch.nn.LogSoftmax(dim=0)(edge_logs[available_indexes]))
            new_edge=all_possible_edges[available_indexes[torch.multinomial(edge_probs,1).item()]]
            edge_list.append(new_edge)

            zpair_r=torch.concat((z[new_edge[0]],z[new_edge[1]]),dim=0)
            r_logs=self.linear_r_2(torch.nn.ReLU()(self.linear_r_1(zpair_r)))
            
            weight_mask=(torch.tensor([1,2,3])<=min(capacities[new_edge[0]],capacities[new_edge[1]]))
            weight_probs=torch.exp(torch.nn.LogSoftmax(dim=0)(r_logs[weight_mask]))
            edge_weights=torch.cat((edge_weights,torch.multinomial(weight_probs,1)+1))

        # Sample s
        edge_list_doubled=edge_list+[(b,a) for a,b in edge_list]
        edge_index=torch.tensor(edge_list_doubled,dtype=torch.long).t().contiguous()
        edge_weights_doubled=torch.cat([edge_weights,edge_weights])

        agg_z=torch.nn.ReLU()(self.GCN_s(z,edge_index,edge_weight=edge_weights_doubled))

        s_mu=self.linear_s_mu_2(torch.nn.ReLU()(self.linear_s_mu_1(agg_z)))
        s_sigma_vect=self.linear_s_sigma_2(torch.nn.ReLU()(self.linear_s_sigma_1(agg_z)))

        s=torch.empty(0)
        for i in range(num_nodes) :
            mu=s_mu[i]
            sigma_vect=s_sigma_vect[i].unsqueeze(0)
            sigma=torch.matmul(sigma_vect.T,sigma_vect)+torch.eye(3)*0.0001
            s=torch.cat((s,torch.distributions.multivariate_normal.MultivariateNormal(mu,sigma).sample().unsqueeze(0)))

        return (t,edge_list,edge_weights,s)