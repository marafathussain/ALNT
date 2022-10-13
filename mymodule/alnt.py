def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)
    
    
    
        
    # 	===== FOLLOWING CODE SNIPPET IS ONLY USABLE AFTER LOADING DATA AND PERFORMING AUGMENTATION =======#
    
    # Defining model and hyperparameters
    device = torch.device('cuda:0')
    model = UNet(dimensions=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2).to(device) # load your own model
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    # model.load_state_dict(torch.load('/home/marafath/scratch/saved_model_iA/mic_f_{}.pth'.format(i))) # load pre-trained weights, if doing fine-tuning

    # Starting a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    epc = 300 # Number of epoch
    for epoch in range(epc):
        print('-' * 10)
        print('epoch {}/{}'.format(epoch + 1, epc))
        model.train()
        epoch_loss = 0
        step = 0

        for noisy_batch_data in noisy_loader:
            step_n = 0
            clean_cycle_loss = 0
            flag = 0
            for clean_batch_data in clean_loader:
                step += 1
                step_n += 1
                meta_net = UNet(dimensions=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2).to(device) # Meta-net, identical to actual model
                meta_net.load_state_dict(model.state_dict())
                meta_net.to(device)
                meta_optimizer = torch.optim.Adam(meta_net.parameters(), 1e-4)

                inputs_n, labels_n = to_var(noisy_batch_data[0],requires_grad=False), to_var(noisy_batch_data[1], requires_grad=False) # loading batch data without expert annotation (but with machine generated annotation)
                y_n_hat  = meta_net(inputs_n)
                labels_n = labels_n.type(torch.float)

                ## Assigning one (Dice + CE) loss per batch item, i.e. image
                labels_n_ce = labels_n.long()
                loss_ce = F.cross_entropy(y_n_hat.squeeze(),labels_n_ce.squeeze(), reduce=False)
                tmp = torch.mean(loss_ce,1)
                tmp = torch.mean(tmp,1)
                loss_ce = torch.mean(tmp,1)

                dice = DiceLoss(to_onehot_y=True, softmax=True, reduction='none')
                loss_dc = dice(y_n_hat,labels_n)
                loss_dc = torch.mean(loss_dc,1)
                
                cost = loss_ce + loss_dc
                g_x = to_var(torch.zeros(cost.size()))
                for b_ in range(noisy_bsize):
                    tmp_cost = cost[b_]
                    tmp_cost.backward(retain_graph=True)
                    tmp_grad = []
            
                    for name, param in meta_net.named_parameters():
                        if name == 'model.2.1.conv.unit0.conv.weight': # last layer of UNet, check the name of the last layer in your model
                            tmp_grad.append(param.grad.view(-1))
                    tmp_grad = torch.cat(tmp_grad)
                    s_ = Variable(torch.sum(torch.abs(tmp_grad)))
                    n_ = tmp_grad.shape[0]
                    with torch.no_grad():
                        g_x[b_] = s_/n_     
                        
                eps = to_var(torch.zeros(cost.size()))
                l_n_meta = torch.sum(cost * eps)
                
                grads_n = torch.autograd.grad(l_n_meta, (meta_net.parameters()), create_graph=True) # for later use

                meta_net.zero_grad()
                l_n_meta.backward() # gradient --> Grad
                meta_optimizer.step() # parameter update: theta_i+1 = theta_i - alpha*Grad

                # 2nd forward pass and getting the gradients with respect to epsilon
                inputs_c, labels_c = to_var(clean_batch_data[0],requires_grad=False), to_var(clean_batch_data[1], requires_grad=False)

                y_c_hat = meta_net(inputs_c)
                labels_c = labels_c.type(torch.float)
                
                # Assigning one (Dice + CE) loss per batch item, i.e. image
                labels_c_ce = labels_c.long()
                loss_ce_c = F.cross_entropy(y_c_hat.squeeze(), labels_c_ce.squeeze())

                dice = DiceLoss(to_onehot_y=True, softmax=True)
                loss_dc_c = dice(y_c_hat, labels_c)
                
                l_c_meta = loss_ce_c + loss_dc_c 
                
                grads_c = torch.autograd.grad(l_c_meta, (meta_net.parameters()), only_inputs=True)
                grad_eps = torch.autograd.grad(grads_n, eps, grads_c, only_inputs=True)[0]

                if flag == 0:
                    temp_grad = grad_eps
                    flag = 1
                    break
                else:
                    temp_grad = temp_grad + grad_eps
                    
            temp_grad_w = temp_grad/step_n       
            
            # Computing and normalizing the weights
            w_tilde = torch.clamp(temp_grad_w, min=0)
            norm_c = torch.sum(w_tilde)
            if norm_c.item() == 0:
                w_e = w_tilde  
            else:
                w_e = w_tilde/norm_c
                   
            norm_c_ = torch.sum(g_x)
            if norm_c_.item() == 0:
                w_x = g_x  
            else:
                w_x = g_x/norm_c_
            
            w = torch.empty(noisy_bsize, requires_grad=False)
            w = alpha*w_e + (1-alpha)*w_x    # alpha = 1 --> RGS; alpha = 0 --> RGM; alpha = 0.5 --> RGS&M
            #w[w < (1/noisy_bsize)] = 0  # --> RGS&M+AL
            
            
            # Computing the loss with the computed weights and then perform a parameter update
            y_n_hat = model(inputs_n)

            # Assigning one (Dice + CE) loss per batch item, i.e. image
            labels_n_ce = labels_n.long()
            loss_ce = F.cross_entropy(y_n_hat.squeeze(),labels_n_ce.squeeze(), reduce=False)
            tmp = torch.mean(loss_ce,1)
            tmp = torch.mean(tmp,1)
            loss_ce = torch.mean(tmp,1)

            dice = DiceLoss(to_onehot_y=True, softmax=True, reduction='none')
            loss_dc = dice(y_n_hat,labels_n)
            loss_dc = torch.mean(loss_dc,1)

            cost = loss_ce + loss_dc
            l_n = torch.sum(cost * w)

            optimizer.zero_grad()
            l_n.backward()
            optimizer.step()

            clean_cycle_loss += l_n.item()
            epoch_loss += l_n.item()
            
            epoch_len = len(noisy_ds) // noisy_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {clean_cycle_loss:.4f}")
            
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print('epoch {} average loss: {:.4f}'.format(epoch + 1, epoch_loss))