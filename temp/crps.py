import torch
import properscoring as ps
import numpy as np
def brown_noise(shape, reddening=2):
    # Parameters that have worked well in practice:
    # reddening ~ 1.2, scale full noise by ~ 0.04 before adding to initial condition
    noise = torch.normal(torch.zeros(shape), torch.ones(shape))
    x_white = torch.fft.fft2(noise)
    S = torch.abs(torch.fft.fftfreq(noise.shape[-2]).reshape(-1, 1))**reddening + torch.abs(torch.fft.fftfreq(noise.shape[-1]))**reddening
                
    S = torch.where(S==0, 0, 1 / S)
    S = S / torch.sqrt(torch.mean(S**2))
    x_shaped = x_white * S
    noise_shaped = torch.fft.ifft2(x_shaped).real
        
    return noise_shaped
  
def get_spread(ens):
    n_members = ens.shape[0]
    s = 0
    mean = ens.mean(axis=0)
    for member in ens:
        s += (mean - member)**2
                
    s /= n_members-1
                
    return np.sqrt(s.mean(axis=(1,2)))

def get_skill_spread(ens, obs):
    ensemble_mean = ens.mean(axis=0)
    MSE = np.mean(((obs - ensemble_mean)**2), axis=(1,2))
    RMSE = np.sqrt(MSE)
    
    spread = get_spread(ens)
    
    return RMSE, spread
  
def ensemble(afno, batch_size, n_members, init_noise_scale, path):
    name = 'big_ensemble_baseline_z500'
    
    ch = 0 # To get CRPS for u10
    
    era5_file = '/pscratch/sd/j/jpathak/34var/out_of_sample/2018.h5'
    ifs_file = f'/global/cfs/cdirs/m4134/tigge_ens_h5/u10/2018_1.h5'  
    with h5py.File(ifs_file, 'r') as f:
        steps = f['era5_steps'][0]
        steps = steps
    stats_path = '/pscratch/sd/j/jpathak/34var/stats/'
    means = torch.tensor(np.load(stats_path + '/global_means.npy')[0,:26,0,0]).cuda()
    stds = torch.tensor(np.load(stats_path + '/global_stds.npy')[0,:26,0,0]).cuda()
    scores = []
    step_count = max(steps)+1
    
    afno.eval()
    for val_example in range(0,5):
        with h5py.File(era5_file, 'r') as g:
            start = val_example * 100
            obs = torch.tensor(g['fields'][start:start+step_count, :26, :720])
            
        with torch.autocast(device_type='cuda'):
            init = obs[0].clone().cuda()
            init -= means[:, None, None]
            init /= stds[:, None, None]
            init = repeat(init, 'ch w h  -> n ch w h', n=n_members)
            init = init + init_noise * brown_noise(init.shape, reddening=1.2).cuda()
            e = torch.zeros_like(init)
            for i in range(1, max(steps[:-1])+1):
                with torch.no_grad():
                    for m in tqdm(range(0, init.shape[0], batch_size)):   
                        init[m:m+batch_size] = afno(init[m:m+batch_size])
                        e[m:m+batch_size] = (init[m:m+batch_size] * stds[:, None, None]) + means[:, None, None]
                    if i in steps:
                        # If your ensembles are small enough you can just get away with the following line. If you have too many members you'll have to split up computation over different areas like below.
                        # crps = ps.crps_ensemble(obs[i, ch].cpu(), e[:, ch].cpu(), axis=0)
                        
                        crps = np.zeros((720, 1440))
                        crps[:360,:720] = ps.crps_ensemble(obs[i, ch, :360,:720].cpu(), e[:, ch, :360,:720].cpu(), axis=0)
                        crps[:360,720:] = ps.crps_ensemble(obs[i, ch, :360,720:].cpu(), e[:, ch, :360,720:].cpu(), axis=0)
                        crps[360:,:720] = ps.crps_ensemble(obs[i, ch, 360:,:720].cpu(), e[:, ch, 360:,:720].cpu(), axis=0)
                        crps[360:,720:] = ps.crps_ensemble(obs[i, ch, 360:,720:].cpu(), e[:, ch, 360:,720:].cpu(), axis=0)
                        rmse, spread = get_skill_spread(e.cpu().numpy(), obs[i].numpy())
                        print(i, crps.mean())
                        scores.append({
                          'lead time (hr)': i * 6,
                          'channel': 0,
                          'CRPS': crps.mean(),
                          'RMSE': rmse[ch],
                          'spread': spread[ch],
                        })
                            
        pd.DataFrame(scores).to_csv(path + 'scores_' + str(val_example) + '.csv')
                                            
def train(afno, optimizer, train_data_loader):
    # Note: This expects the train_data_loader batch size to be 1.
    # The actual batch size then becomes n_members. We can then reduce variance by using gradient accumulation to simulate larger batches.
  
    n_members = 16 # Try how large you can go here without running OOM.
    init_noise = 0.1 # You'll have to try what works well for you here.
    accum_iter = 4
  
    for batch_idx, data in enumerate(train_data_loader, 0):
        afno.train()
        inp, tar = map(lambda x: torch.squeeze(x.to(device, dtype=torch.float32), dim=1), data)
        # Copy initial condition and add noise
        init = repeat(inp, 'b c h w -> (b n) c h w', n=n_members)
        init = init + init_noise * brown_noise(init.shape, reddening=1.2).cuda()
        # train_data_loader needs to be multistep for this to make sense
        for ts, y in enumerate(tar[0]):
            # Do forward pass of network
            with torch.autocast(device_type='cuda'):
                forecast = afno(init, ts)
                
                x = rearrange(forecast, '(b n) c h w -> b n (c h w)', n=n_members)
                y = rearrange(y.unsqueeze(0), 'b c h w -> b (c h w)')
            # Calculate Energy Score (loss)
            batch_size, ensemble_size, data_size = x.shape
            ensemble_loss = 0
            for x_i in x[0]:
                # for each enemble member, calculate the score of that member against all others
                with torch.autocast(device_type='cuda'):
                    x_i = x_i.unsqueeze(0).unsqueeze(0)
                    diff_X_y = torch.cdist(y.reshape(batch_size, 1, data_size), x_i, p=2)
                    diff_X_y = torch.squeeze(diff_X_y, dim=1)
                    diff_X_tildeX = torch.cdist(x, x_i, p=2)
                    t1 = 2 * torch.mean(diff_X_y, dim=1) 
                    t2 = torch.sum(diff_X_tildeX) / ((ensemble_size - 1))
                    loss = t1 - t2
                    loss = loss / ensemble_size
                    loss = loss / (len(tar[0]) * accum_iter)
                    
                loss.backward(retain_graph=True)
                ensemble_loss += loss.item()
                del x_i
                del t1
                del t2
                del diff_X_y
                del diff_X_tildeX
            loss_mean += loss.item()
            losses_mean[ts] += loss.item()
            init = forecast.detach()
            del forecast
            del loss
            del x
            del y
        
        if (batch_idx + 1) % accum_iter == 0:
            #torch.nn.utils.clip_grad_norm_(afno.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            print(loss_mean)
            print(losses_mean)
            loss_mean = 0.0
            losses_mean = np.zeros(20)
