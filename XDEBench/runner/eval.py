import os
import torch
import time 
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from evaluator.metrics import *
from utils.tools import *
from data.graph_dataset import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"

def evalModel_deeponet(args, net, dataset, test_data, coords, times, num_chemical, writer, step, finalFlag=False, varlist=None, save_raw=False, raw_out_dir=None):
    device = args.device
    net.eval()
    steps = test_data.shape[1] - 1

    num_chemical = dataset["num_chemical"]
    num_temperature = dataset["num_temperature"]
    num_density = dataset["num_density"]
    num_velocity = dataset["num_velocity"]
    num_pressure = dataset["num_pressure"]

    with torch.no_grad():
        test_data = test_data.to(device)
        total_losses = dict(chemical=0.0, temperature=0.0, density=0.0, velocity=0.0, pressure=0.0)

        MeanPSNR = 0.0
        MeanSSIM = 0.0
        MeanMSSSIM = 0.0
        MeanGMSD = 0.0
        MeanMSGMSD = 0.0

        ### inference step by step
        inference_time_total = 0.0
        times = times.repeat(test_data.shape[0], 1, 1, 1)
        for t in range(steps):
            inference_start = time.time()
            if t == 0:
                input_data = net.encoder(test_data[:, 0])

            input_time = times[:, t + 1].unsqueeze(1)  # [batch, 1, H, W]
            output_pred = net(input_data, coords, input_time)
            output_gth = net.encoder(test_data[:, t + 1])

            inference_end = time.time()
            inference_time = inference_end - inference_start
            inference_time_total += inference_time

            diff = output_pred - output_gth
            total_losses['chemical'] += (diff[:, :num_chemical] ** 2).mean().item()
            total_losses['temperature'] += (diff[:, num_chemical:num_chemical + num_temperature] ** 2).mean().item()
            if num_density > 0:
                total_losses['density'] += (diff[:, num_chemical + num_temperature:num_chemical + num_temperature + num_density] ** 2).mean().item()
            total_losses['velocity'] += (diff[:, num_chemical + num_temperature + num_density:num_chemical + num_temperature + num_density + num_velocity] ** 2).mean().item()
            if num_pressure > 0:
                total_losses['pressure'] += (diff[:, num_chemical + num_temperature + num_density + num_velocity:] ** 2).mean().item()
                
            if finalFlag and save_raw and raw_out_dir is not None:
                pred_denorm = net.decoder(output_pred.clone())   # [B,C,H,W]
                gt_denorm   = net.decoder(output_gth.clone())    # [B,C,H,W]

                B = pred_denorm.shape[0]
                raw_coords_np = np.array(dataset["coords"])  # 原始坐标（保持原样）

                for iCase in range(B):
                    pred_np = pred_denorm[iCase].detach().cpu().numpy()  # [C,H,W]
                    gt_np   = gt_denorm[iCase].detach().cpu().numpy()    # [C,H,W]

                    save_path = os.path.join(raw_out_dir, f"case{iCase}_t{t+1:04d}.npz")
                    np.savez_compressed(
                        save_path,
                        pred=pred_np,
                        gt=gt_np,
                        vars=np.array(varlist, dtype=object),
                        coords=raw_coords_np,
                        time_index=t+1
                    )
                    
            if finalFlag:    
                for iCase in range(output_gth.shape[0]):
                    for iC in range(output_gth.shape[1]):
                        pred_img = output_pred[iCase, iC]
                        gth_img = output_gth[iCase, iC]

                        colored_pred_img, colored_gth_img = apply_colormap(pred_img, gth_img, cmap='jet')

                        writer.add_image(f"Traj{iCase}_{varlist[iC][:-4]}_image/{t+1}_pred", colored_pred_img, step)
                        writer.add_image(f"Traj{iCase}_{varlist[iC][:-4]}_image/{t+1}_gt", colored_gth_img, step) 

            # Get Metric
            PSNR = 0.0
            SSIM = 0.0
            MSSSIM = 0.0
            GMSD = 0.0
            MSGMSD = 0.0

            for iC in range(output_gth.shape[1]):
                pred_img = output_pred[:, iC:iC+1, ...]
                gth_img = output_gth[:, iC:iC+1, ...]

                PSNR += calculate_psnr(gth_img, pred_img)
                SSIM += calculate_ssim(gth_img, pred_img)
                MSSSIM += calculate_ms_ssim(gth_img, pred_img)
                GMSD += calculate_gmsd(gth_img, pred_img)
                MSGMSD += calculate_ms_gmsd(gth_img, pred_img)

            PSNR /= output_gth.shape[1]
            SSIM /= output_gth.shape[1]
            MSSSIM /= output_gth.shape[1]
            GMSD /= output_gth.shape[1]
            MSGMSD /= output_gth.shape[1]

            MeanPSNR += PSNR
            MeanSSIM += SSIM
            MeanMSSSIM += MSSSIM
            MeanGMSD += GMSD
            MeanMSGMSD += MSGMSD
        
        MeanPSNR /= steps
        MeanSSIM /= steps
        MeanMSSSIM /= steps
        MeanGMSD /= steps
        MeanMSGMSD /= steps

        ### inference all steps at once
        # inference_start = time.time()

        # init_data = net.encoder(test_data[:, 0])
        # batch_size = init_data.shape[0]
        # input_data = init_data.repeat(steps, 1, 1, 1)
        
        # times = times.repeat(init_data.shape[0], 1, 1, 1)
        # input_time = times[:, 1:].permute(1, 0, 2, 3)  # [steps, batch, 1, H, W]
        # input_time = input_time.reshape(-1, 1, input_time.shape[-2], input_time.shape[-1])  # [steps*batch, 1, H, W]
        
        # output_pred = net(input_data, coords, input_time)
        
        # output_pred = output_pred.reshape(steps, batch_size, *output_pred.shape[1:])
        # output_pred = output_pred.permute(1, 0, 2, 3, 4)  # [batch, steps, C, H, W]

        # inference_end = time.time()
        # inference_time_total = inference_end - inference_start

        # output_gth = torch.stack([net.encoder(test_data[:, i]) for i in range(1, steps+1)], dim=1)
        
        # diff = (output_pred - output_gth)**2 

        # # 定义辅助函数：steps维度求和 + 其他维度平均
        # def channel_mean_with_steps_sum(tensor):
        #     return tensor.sum(dim=1).mean().item()

        # # 计算各物理量损失
        # total_losses['chemical'] = channel_mean_with_steps_sum(diff[:, :, :num_chemical])
        # total_losses['temperature'] = channel_mean_with_steps_sum(diff[:, :, num_chemical:num_chemical+num_temperature])
        # total_losses['density'] = channel_mean_with_steps_sum(diff[:, :, num_chemical+num_temperature:num_chemical+num_temperature+num_density])
        # total_losses['velocity'] = channel_mean_with_steps_sum(diff[:, :, num_chemical+num_temperature+num_density:num_chemical+num_temperature+num_density+num_velocity])
        # if num_pressure > 0:
        #     total_losses['pressure'] = channel_mean_with_steps_sum(diff[:, :, num_chemical+num_temperature+num_density+num_velocity:])
        
        # if finalFlag:
        #     for t in range(steps):
        #         for iC in range(output_gth.shape[2]):
        #             pred_img = output_pred[0, t, iC]
        #             gth_img = output_gth[0, t, iC]
                    
        #             colored_pred_img, colored_gth_img = apply_colormap(pred_img, gth_img, cmap='jet')
                    
        #             writer.add_image(f"{varlist[iC][:-4]}_image/{t+1}_pred", colored_pred_img, step)
        #             writer.add_image(f"{varlist[iC][:-4]}_image/{t+1}_gt", colored_gth_img, step)

        # # Get Metric
        # MeanPSNR = 0.0
        # MeanSSIM = 0.0
        # MeanMSSSIM = 0.0
        # for t in range(steps):
        #     PSNR = 0.0
        #     SSIM = 0.0
        #     MSSSIM = 0.0
        #     for iC in range(output_gth.shape[2]):
        #         pred_img = output_pred[:, t, iC:iC+1]
        #         gth_img = output_gth[:, t, iC:iC+1]
        #         PSNR += calculate_psnr(gth_img, pred_img)
        #         SSIM += calculate_ssim(gth_img, pred_img)
        #         MSSSIM += calculate_ms_ssim(gth_img, pred_img)

        #     PSNR /= output_gth.shape[1]
        #     SSIM /= output_gth.shape[1]
        #     MSSSIM /= output_gth.shape[1]

        #     MeanPSNR += PSNR
        #     MeanSSIM += SSIM
        #     MeanMSSSIM += MSSSIM
        
        # MeanPSNR /= steps
        # MeanSSIM /= steps
        # MeanMSSSIM /= steps

        total_loss = sum(total_losses.values())
        for k, v in total_losses.items():
            print(f"  {k.capitalize():<12}: {v:.6f}")
        print(f"  TOTAL        : {total_loss:.6f}")
        return total_loss, MeanPSNR, MeanSSIM, MeanMSSSIM, MeanGMSD, MeanMSGMSD, inference_time_total
    
def evalModel_deeponet_3d(args, net, dataset, test_data, coords, times, num_chemical, writer, step, finalFlag=False, varlist=None, save_raw=False, raw_out_dir=None):
    device = args.device
    net.eval()
    steps = test_data.shape[1] - 1
    num_train, T, C, x_size, y_size, z_size = test_data.shape

    num_chemical = dataset["num_chemical"]
    num_temperature = dataset["num_temperature"]
    num_density = dataset["num_density"]
    num_velocity = dataset["num_velocity"]
    num_pressure = dataset["num_pressure"]

    with torch.no_grad():
        # test_data = test_data.to(device)
        total_losses = dict(chemical=0.0, temperature=0.0, density=0.0, velocity=0.0, pressure=0.0)

        MeanPSNR = 0.0
        MeanSSIM = 0.0
        MeanMSSSIM = 0.0
        MeanGMSD = 0.0
        MeanMSGMSD = 0.0

        ### inference step by step
        inference_time_total = 0.0
        times = times.repeat(test_data.shape[0], 1, 1, 1, 1)
        for t in range(steps):
            inference_start = time.time()
            if t == 0:
                input_data = net.encoder(test_data[:, 0].to(device))

            input_time = times[:, t + 1].unsqueeze(1).to(device)  # [batch, 1, H, W]
            output_pred = net(input_data, coords, input_time)
            output_gth = net.encoder(test_data[:, t + 1].to(device))

            inference_end = time.time()
            inference_time = inference_end - inference_start
            inference_time_total += inference_time

            diff = output_pred - output_gth
            total_losses['chemical'] += (diff[:, :num_chemical] ** 2).mean().item()
            total_losses['temperature'] += (diff[:, num_chemical:num_chemical + num_temperature] ** 2).mean().item()
            if num_density > 0:
                total_losses['density'] += (diff[:, num_chemical + num_temperature:num_chemical + num_temperature + num_density] ** 2).mean().item()
            total_losses['velocity'] += (diff[:, num_chemical + num_temperature + num_density:num_chemical + num_temperature + num_density + num_velocity] ** 2).mean().item()
            if num_pressure > 0:
                total_losses['pressure'] += (diff[:, num_chemical + num_temperature + num_density + num_velocity:] ** 2).mean().item()
                
            if finalFlag and save_raw and raw_out_dir is not None:
                pred_denorm = net.decoder(output_pred.clone())   # [B,C,H,W]
                gt_denorm   = net.decoder(output_gth.clone())    # [B,C,H,W]

                B = pred_denorm.shape[0]
                raw_coords_np = np.array(dataset["coords"])  # 原始坐标（保持原样）

                for iCase in range(B):
                    pred_np = pred_denorm[iCase].detach().cpu().numpy()  # [C,H,W]
                    gt_np   = gt_denorm[iCase].detach().cpu().numpy()    # [C,H,W]

                    save_path = os.path.join(raw_out_dir, f"case{iCase}_t{t+1:04d}.npz")
                    np.savez_compressed(
                        save_path,
                        pred=pred_np,
                        gt=gt_np,
                        vars=np.array(varlist, dtype=object),
                        coords=raw_coords_np,
                        time_index=t+1
                    )
                    
            if finalFlag:
                for iCase in range(output_gth.shape[0]):
                    for iC in range(output_gth.shape[1]):
                        predx_img = output_pred[iCase, iC, int(x_size/2), :, :]
                        gthx_img = output_gth[iCase, iC, int(x_size/2), :, :]

                        colored_predx_img, colored_gthx_img = apply_colormap(predx_img, gthx_img, cmap='jet')

                        writer.add_image(f"Traj{iCase}_{varlist[iC][:-4]}_image/{t+1}_predx", colored_predx_img, step)
                        writer.add_image(f"Traj{iCase}_{varlist[iC][:-4]}_image/{t+1}_gtx", colored_gthx_img, step)

                        predy_img = output_pred[iCase, iC, :, int(y_size/2), :]
                        gthy_img = output_gth[iCase, iC, :, int(y_size/2), :]

                        colored_predy_img, colored_gthy_img = apply_colormap(predy_img, gthy_img, cmap='jet')
                        
                        writer.add_image(f"Traj{iCase}_{varlist[iC][:-4]}_image/{t+1}_predy", colored_predy_img, step)
                        writer.add_image(f"Traj{iCase}_{varlist[iC][:-4]}_image/{t+1}_gty", colored_gthy_img, step)

                        predz_img = output_pred[iCase, iC, :, :, int(z_size/2)]
                        gthz_img = output_gth[iCase, iC, :, :, int(z_size/2)]

                        colored_predz_img, colored_gthz_img = apply_colormap(predz_img, gthz_img, cmap='jet')
                        
                        writer.add_image(f"Traj{iCase}_{varlist[iC][:-4]}_image/{t+1}_predz", colored_predz_img, step)
                        writer.add_image(f"Traj{iCase}_{varlist[iC][:-4]}_image/{t+1}_gtz", colored_gthz_img, step) 

            # Get Metric
            PSNR = 0.0
            SSIM = 0.0
            MSSSIM = 0.0
            GMSD = 0.0
            MSGMSD = 0.0

            for iC in range(output_gth.shape[1]):
                # x-direction
                pred_img = output_pred[:, iC:iC+1, int(x_size/2), :, :]
                gth_img = output_gth[:, iC:iC+1, int(x_size/2), :, :]

                PSNR += calculate_psnr(gth_img, pred_img)
                SSIM += calculate_ssim(gth_img, pred_img)
                MSSSIM += calculate_ms_ssim(gth_img, pred_img)
                GMSD += calculate_gmsd(gth_img, pred_img)
                MSGMSD += calculate_ms_gmsd(gth_img, pred_img)

                # y-direction
                pred_img = output_pred[:, iC:iC+1, :, int(y_size/2), :]
                gth_img = output_gth[:, iC:iC+1, :, int(y_size/2), :]

                PSNR += calculate_psnr(gth_img, pred_img)
                SSIM += calculate_ssim(gth_img, pred_img)
                MSSSIM += calculate_ms_ssim(gth_img, pred_img)
                GMSD += calculate_gmsd(gth_img, pred_img)
                MSGMSD += calculate_ms_gmsd(gth_img, pred_img)

                # z-direction
                pred_img = output_pred[:, iC:iC+1, :, :, int(z_size/2)]
                gth_img = output_gth[:, iC:iC+1, :, :, int(z_size/2)]

                PSNR += calculate_psnr(gth_img, pred_img)
                SSIM += calculate_ssim(gth_img, pred_img)
                MSSSIM += calculate_ms_ssim(gth_img, pred_img)
                GMSD += calculate_gmsd(gth_img, pred_img)
                MSGMSD += calculate_ms_gmsd(gth_img, pred_img)


            PSNR /= output_gth.shape[1] * 3
            SSIM /= output_gth.shape[1] * 3
            MSSSIM /= output_gth.shape[1] * 3
            GMSD /= output_gth.shape[1] * 3
            MSGMSD /= output_gth.shape[1] * 3

            MeanPSNR += PSNR
            MeanSSIM += SSIM
            MeanMSSSIM += MSSSIM
            MeanGMSD += GMSD
            MeanMSGMSD += MSGMSD
        
        MeanPSNR /= steps
        MeanSSIM /= steps
        MeanMSSSIM /= steps
        MeanGMSD /= steps
        MeanMSGMSD /= steps

        total_loss = sum(total_losses.values())
        for k, v in total_losses.items():
            print(f"  {k.capitalize():<12}: {v:.6f}")
        print(f"  TOTAL        : {total_loss:.6f}")
        return total_loss, MeanPSNR, MeanSSIM, MeanMSSSIM, MeanGMSD, MeanMSGMSD, inference_time_total
    
def evalModel(args, net, dataset, test_data, coords, num_chemical, writer, step, finalFlag=False, varlist=None, save_raw=False, raw_out_dir=None):
    device = args.device
    net.eval()
    steps = test_data.shape[1] - 1

    num_chemical = dataset["num_chemical"]
    num_temperature = dataset["num_temperature"]
    num_density = dataset["num_density"]
    num_velocity = dataset["num_velocity"]
    num_pressure = dataset["num_pressure"]

    with torch.no_grad():
        test_data = test_data.to(device)
        total_losses = dict(chemical=0.0, temperature=0.0, density=0.0, velocity=0.0, pressure=0.0)
        MeanPSNR = 0.0
        MeanSSIM = 0.0
        MeanMSSSIM = 0.0
        MeanGMSD = 0.0
        MeanMSGMSD = 0.0

        inference_time_total = 0.0
        for t in range(steps):
            inference_start = time.time()
            if t == 0:
                input_data = net.encoder(test_data[:, 0])

            output_pred = net(input_data, coords, None)
            output_gth = net.encoder(test_data[:, t + 1])

            input_data = output_pred

            inference_end = time.time()
            inference_time = inference_end - inference_start
            inference_time_total += inference_time

            diff = output_pred - output_gth
            total_losses['chemical'] += (diff[:, :num_chemical] ** 2).mean().item()
            total_losses['temperature'] += (diff[:, num_chemical:num_chemical + num_temperature] ** 2).mean().item()
            if num_density > 0:
                total_losses['density'] += (diff[:, num_chemical + num_temperature:num_chemical + num_temperature + num_density] ** 2).mean().item()
            total_losses['velocity'] += (diff[:, num_chemical + num_temperature + num_density:num_chemical + num_temperature + num_density + num_velocity] ** 2).mean().item()
            if num_pressure > 0:
                total_losses['pressure'] += (diff[:, num_chemical + num_temperature + num_density + num_velocity:] ** 2).mean().item()
            
            if finalFlag and save_raw and raw_out_dir is not None:
                pred_denorm = net.decoder(output_pred.clone())   # [B,C,H,W]
                gt_denorm   = net.decoder(output_gth.clone())    # [B,C,H,W]

                B = pred_denorm.shape[0]
                raw_coords_np = np.array(dataset["coords"])  # 原始坐标（保持原样）

                for iCase in range(B):
                    pred_np = pred_denorm[iCase].detach().cpu().numpy()  # [C,H,W]
                    gt_np   = gt_denorm[iCase].detach().cpu().numpy()    # [C,H,W]

                    save_path = os.path.join(raw_out_dir, f"case{iCase}_t{t+1:04d}.npz")
                    np.savez_compressed(
                        save_path,
                        pred=pred_np,
                        gt=gt_np,
                        vars=np.array(varlist, dtype=object),
                        coords=raw_coords_np,
                        time_index=t+1
                    )

            if finalFlag:       
                for iCase in range(output_gth.shape[0]):
                    for iC in range(output_gth.shape[1]):
                        pred_img = output_pred[iCase, iC]
                        gth_img = output_gth[iCase, iC]

                        colored_pred_img, colored_gth_img = apply_colormap(pred_img, gth_img, cmap='jet')

                        writer.add_image(f"Traj{iCase}_{varlist[iC][:-4]}_image/{t+1}_pred", colored_pred_img, step)
                        writer.add_image(f"Traj{iCase}_{varlist[iC][:-4]}_image/{t+1}_gt", colored_gth_img, step)

            # Get Metric
            PSNR = 0.0
            SSIM = 0.0
            MSSSIM = 0.0
            GMSD = 0.0
            MSGMSD = 0.0

            for iC in range(output_gth.shape[1]):
                pred_img = output_pred[:, iC:iC+1, ...]
                gth_img = output_gth[:, iC:iC+1, ...]

                PSNR += calculate_psnr(gth_img, pred_img)
                SSIM += calculate_ssim(gth_img, pred_img)
                MSSSIM += calculate_ms_ssim(gth_img, pred_img)
                GMSD += calculate_gmsd(gth_img, pred_img)
                MSGMSD += calculate_ms_gmsd(gth_img, pred_img)

            PSNR /= output_gth.shape[1]
            SSIM /= output_gth.shape[1]
            MSSSIM /= output_gth.shape[1]
            GMSD /= output_gth.shape[1]
            MSGMSD /= output_gth.shape[1]

            MeanPSNR += PSNR
            MeanSSIM += SSIM
            MeanMSSSIM += MSSSIM
            MeanGMSD += GMSD
            MeanMSGMSD += MSGMSD
        
        MeanPSNR /= steps
        MeanSSIM /= steps
        MeanMSSSIM /= steps
        MeanGMSD /= steps
        MeanMSGMSD /= steps

        total_loss = sum(total_losses.values())
        for k, v in total_losses.items():
            print(f"  {k.capitalize():<12}: {v:.6f}")
        print(f"  TOTAL        : {total_loss:.6f}")
        return total_loss, MeanPSNR, MeanSSIM, MeanMSSSIM, MeanGMSD, MeanMSGMSD, inference_time_total
    
def evalModel_3d(args, net, dataset, test_data, coords, num_chemical, writer, step, finalFlag=False, varlist=None, save_raw=False, raw_out_dir=None):
    device = args.device
    net.eval()
    steps = test_data.shape[1] - 1
    num_train, T, C, x_size, y_size, z_size = test_data.shape
    crop_size = args.crop_size
    overlap_size = args.overlap_size

    num_chemical = dataset["num_chemical"]
    num_temperature = dataset["num_temperature"]
    num_density = dataset["num_density"]
    num_velocity = dataset["num_velocity"]
    num_pressure = dataset["num_pressure"]

    with torch.no_grad():
        test_data = test_data.to(device)
        total_losses = dict(chemical=0.0, temperature=0.0, density=0.0, velocity=0.0, pressure=0.0)
        MeanPSNR = 0.0
        MeanSSIM = 0.0
        MeanMSSSIM = 0.0
        MeanGMSD = 0.0
        MeanMSGMSD = 0.0

        inference_time_total = 0.0
        for t in range(steps):
            inference_start = time.time()
            if t == 0:
                input_data = net.encoder(test_data[:, 0])

            output_pred = net(input_data, coords, None)
            output_gth = net.encoder(test_data[:, t + 1])

            input_data = output_pred

            inference_end = time.time()
            inference_time = inference_end - inference_start
            inference_time_total += inference_time
            
            diff = output_pred - output_gth
            total_losses['chemical'] += (diff[:, :num_chemical] ** 2).mean().item()
            total_losses['temperature'] += (diff[:, num_chemical:num_chemical + num_temperature] ** 2).mean().item()
            if num_density > 0:
                total_losses['density'] += (diff[:, num_chemical + num_temperature:num_chemical + num_temperature + num_density] ** 2).mean().item()
            total_losses['velocity'] += (diff[:, num_chemical + num_temperature + num_density:num_chemical + num_temperature + num_density + num_velocity] ** 2).mean().item()
            if num_pressure > 0:
                total_losses['pressure'] += (diff[:, num_chemical + num_temperature + num_density + num_velocity:] ** 2).mean().item()
                
            if finalFlag and save_raw and raw_out_dir is not None:
                pred_denorm = net.decoder(output_pred.clone())   # [B,C,H,W]
                gt_denorm   = net.decoder(output_gth.clone())    # [B,C,H,W]

                B = pred_denorm.shape[0]
                raw_coords_np = np.array(dataset["coords"])  # 原始坐标（保持原样）

                for iCase in range(B):
                    pred_np = pred_denorm[iCase].detach().cpu().numpy()  # [C,H,W]
                    gt_np   = gt_denorm[iCase].detach().cpu().numpy()    # [C,H,W]

                    save_path = os.path.join(raw_out_dir, f"case{iCase}_t{t+1:04d}.npz")
                    np.savez_compressed(
                        save_path,
                        pred=pred_np,
                        gt=gt_np,
                        vars=np.array(varlist, dtype=object),
                        coords=raw_coords_np,
                        time_index=t+1
                    )
                    
            if finalFlag:
                for iCase in range(output_gth.shape[0]):
                    for iC in range(output_gth.shape[1]):
                        predx_img = output_pred[iCase, iC, int(x_size/2), :, :]
                        gthx_img = output_gth[iCase, iC, int(x_size/2), :, :]

                        colored_predx_img, colored_gthx_img = apply_colormap(predx_img, gthx_img, cmap='jet')

                        writer.add_image(f"Traj{iCase}_{varlist[iC][:-4]}_image/{t+1}_predx", colored_predx_img, step)
                        writer.add_image(f"Traj{iCase}_{varlist[iC][:-4]}_image/{t+1}_gtx", colored_gthx_img, step)

                        predy_img = output_pred[iCase, iC, :, int(y_size/2), :]
                        gthy_img = output_gth[iCase, iC, :, int(y_size/2), :]

                        colored_predy_img, colored_gthy_img = apply_colormap(predy_img, gthy_img, cmap='jet')
                        
                        writer.add_image(f"Traj{iCase}_{varlist[iC][:-4]}_image/{t+1}_predy", colored_predy_img, step)
                        writer.add_image(f"Traj{iCase}_{varlist[iC][:-4]}_image/{t+1}_gty", colored_gthy_img, step)

                        predz_img = output_pred[iCase, iC, :, :, int(z_size/2)]
                        gthz_img = output_gth[iCase, iC, :, :, int(z_size/2)]

                        colored_predz_img, colored_gthz_img = apply_colormap(predz_img, gthz_img, cmap='jet')
                        
                        writer.add_image(f"Traj{iCase}_{varlist[iC][:-4]}_image/{t+1}_predz", colored_predz_img, step)
                        writer.add_image(f"Traj{iCase}_{varlist[iC][:-4]}_image/{t+1}_gtz", colored_gthz_img, step) 

            # Get Metric
            PSNR = 0.0
            SSIM = 0.0
            MSSSIM = 0.0
            GMSD = 0.0
            MSGMSD = 0.0

            for iC in range(output_gth.shape[1]):
                # x-direction
                pred_img = output_pred[:, iC:iC+1, int(x_size/2), :, :]
                gth_img = output_gth[:, iC:iC+1, int(x_size/2), :, :]

                PSNR += calculate_psnr(gth_img, pred_img)
                SSIM += calculate_ssim(gth_img, pred_img)
                MSSSIM += calculate_ms_ssim(gth_img, pred_img)
                GMSD += calculate_gmsd(gth_img, pred_img)
                MSGMSD += calculate_ms_gmsd(gth_img, pred_img)

                # y-direction
                pred_img = output_pred[:, iC:iC+1, :, int(y_size/2), :]
                gth_img = output_gth[:, iC:iC+1, :, int(y_size/2), :]

                PSNR += calculate_psnr(gth_img, pred_img)
                SSIM += calculate_ssim(gth_img, pred_img)
                MSSSIM += calculate_ms_ssim(gth_img, pred_img)
                GMSD += calculate_gmsd(gth_img, pred_img)
                MSGMSD += calculate_ms_gmsd(gth_img, pred_img)

                # z-direction
                pred_img = output_pred[:, iC:iC+1, :, :, int(z_size/2)]
                gth_img = output_gth[:, iC:iC+1, :, :, int(z_size/2)]

                PSNR += calculate_psnr(gth_img, pred_img)
                SSIM += calculate_ssim(gth_img, pred_img)
                MSSSIM += calculate_ms_ssim(gth_img, pred_img)
                GMSD += calculate_gmsd(gth_img, pred_img)
                MSGMSD += calculate_ms_gmsd(gth_img, pred_img)


            PSNR /= output_gth.shape[1] * 3
            SSIM /= output_gth.shape[1] * 3
            MSSSIM /= output_gth.shape[1] * 3
            GMSD /= output_gth.shape[1] * 3
            MSGMSD /= output_gth.shape[1] * 3

            MeanPSNR += PSNR
            MeanSSIM += SSIM
            MeanMSSSIM += MSSSIM
            MeanGMSD += GMSD
            MeanMSGMSD += MSGMSD
        
        MeanPSNR /= steps
        MeanSSIM /= steps
        MeanMSSSIM /= steps
        MeanGMSD /= steps
        MeanMSGMSD /= steps

        total_loss = sum(total_losses.values())
        for k, v in total_losses.items():
            print(f"  {k.capitalize():<12}: {v:.6f}")
        print(f"  TOTAL        : {total_loss:.6f}")
        return total_loss, MeanPSNR, MeanSSIM, MeanMSSSIM, MeanGMSD, MeanMSGMSD, inference_time_total
    
def evalModel_deeponetU(args, net, dataset, test_data, coords, times, num_chemical, writer, step, finalFlag=False, varlist=None, save_raw=False, raw_out_dir=None):
    device = args.device
    net.eval()
    steps = test_data.shape[1] - 1
    num_channels = test_data.shape[2]

    num_chemical = dataset["num_chemical"]
    num_temperature = dataset["num_temperature"]
    num_density = dataset["num_density"]
    num_velocity = dataset["num_velocity"]
    num_pressure = dataset["num_pressure"]

    with torch.no_grad():
        test_data = test_data.to(device)
        total_losses = dict(chemical=0.0, temperature=0.0, density=0.0, velocity=0.0, pressure=0.0)

        MeanPSNR = 0.0
        MeanSSIM = 0.0
        MeanMSSSIM = 0.0
        MeanGMSD = 0.0
        MeanMSGMSD = 0.0

        ### inference step by step
        inference_time_total = 0.0
        times = times.repeat(test_data.shape[0], 1, 1)
        for t in range(steps):
            inference_start = time.time()
            if t == 0:
                input_data = net.encoder(test_data[:, 0])

            input_time = times[:, t + 1].unsqueeze(1)  # [batch, 1, H, W]
            output_pred = net(input_data, coords, input_time)
            output_gth = net.encoder(test_data[:, t + 1])

            inference_end = time.time()
            inference_time = inference_end - inference_start
            inference_time_total += inference_time

            diff = output_pred - output_gth
            total_losses['chemical'] += (diff[:, :num_chemical] ** 2).mean().item()
            total_losses['temperature'] += (diff[:, num_chemical:num_chemical + num_temperature] ** 2).mean().item()
            if num_density > 0:
                total_losses['density'] += (diff[:, num_chemical + num_temperature:num_chemical + num_temperature + num_density] ** 2).mean().item()
            total_losses['velocity'] += (diff[:, num_chemical + num_temperature + num_density:num_chemical + num_temperature + num_density + num_velocity] ** 2).mean().item()
            if num_pressure > 0:
                total_losses['pressure'] += (diff[:, num_chemical + num_temperature + num_density + num_velocity:] ** 2).mean().item()
                
            if finalFlag and save_raw and raw_out_dir is not None:
                pred_denorm = net.decoder(output_pred.clone())   # [B,C,H,W]
                gt_denorm   = net.decoder(output_gth.clone())    # [B,C,H,W]

                B = pred_denorm.shape[0]
                raw_coords_np = np.array(dataset["coords"])  # 原始坐标（保持原样）

                for iCase in range(B):
                    pred_np = pred_denorm[iCase].detach().cpu().numpy()  # [C,H,W]
                    gt_np   = gt_denorm[iCase].detach().cpu().numpy()    # [C,H,W]

                    save_path = os.path.join(raw_out_dir, f"case{iCase}_t{t+1:04d}.npz")
                    np.savez_compressed(
                        save_path,
                        pred=pred_np,
                        gt=gt_np,
                        vars=np.array(varlist, dtype=object),
                        coords=raw_coords_np,
                        time_index=t+1
                    )
                    
            if finalFlag:
                for iCase in range(test_data.shape[0]):
                        
                    # Get Metric
                    PSNR = 0.0
                    SSIM = 0.0
                    MSSSIM = 0.0
                    GMSD = 0.0
                    MSGMSD = 0.0
                    for iC in range(num_channels):
                        pred_data = output_pred[iCase, iC, :].detach().cpu().numpy()
                        gt_data = output_gth[iCase, iC, :].detach().cpu().numpy()
                        x_coords = coords[0, 0, :].detach().cpu().numpy()
                        y_coords = coords[0, 1, :].detach().cpu().numpy()
                        vmin = min(gt_data.min(), pred_data.min())
                        vmax = max(gt_data.max(), pred_data.max())

                        fig_width = 8
                        fig_height = 1

                        gt_img_array = create_grayscale_scatter(x_coords, y_coords, gt_data, 
                            vmin=vmin, vmax=vmax, figsize=(fig_width, fig_height))
                        pred_img_array = create_grayscale_scatter(x_coords, y_coords, pred_data, 
                            vmin=vmin, vmax=vmax, figsize=(fig_width, fig_height))
                        
                        colored_pred_img, colored_gth_img = apply_colormap(pred_img_array, gt_img_array, cmap='jet')
                        
                        writer.add_image(f"Traj{iCase}_{varlist[iC][:-4]}_image/{t+1}_GT", colored_gth_img, global_step=0)
                        writer.add_image(f"Traj{iCase}_{varlist[iC][:-4]}_image/{t+1}_Pred", colored_pred_img, global_step=0)

                        pred_img = torch.from_numpy(pred_img_array).float().unsqueeze(0) / 255.0
                        gth_img = torch.from_numpy(gt_img_array).float().unsqueeze(0) / 255.0
                        pred_img = pred_img.unsqueeze(0)
                        gth_img = gth_img.unsqueeze(0)

                        PSNR += calculate_psnr(gth_img, pred_img)
                        SSIM += calculate_ssim(gth_img, pred_img)
                        MSSSIM += calculate_ms_ssim(gth_img, pred_img)
                        GMSD += calculate_gmsd(gth_img, pred_img)
                        MSGMSD += calculate_ms_gmsd(gth_img, pred_img)

                    PSNR /= output_gth.shape[1]
                    SSIM /= output_gth.shape[1]
                    MSSSIM /= output_gth.shape[1]
                    GMSD /= output_gth.shape[1]
                    MSGMSD /= output_gth.shape[1]

                    MeanPSNR += PSNR
                    MeanSSIM += SSIM
                    MeanMSSSIM += MSSSIM
                    MeanGMSD += GMSD
                    MeanMSGMSD += MSGMSD
        
        MeanPSNR /= steps * test_data.shape[0]
        MeanSSIM /= steps * test_data.shape[0]
        MeanMSSSIM /= steps * test_data.shape[0]
        MeanGMSD /= steps * test_data.shape[0]
        MeanMSGMSD /= steps * test_data.shape[0]

        total_loss = sum(total_losses.values())
        for k, v in total_losses.items():
            print(f"  {k.capitalize():<12}: {v:.6f}")
        print(f"  TOTAL        : {total_loss:.6f}")
        return total_loss, MeanPSNR, MeanSSIM, MeanMSSSIM, MeanGMSD, MeanMSGMSD, inference_time_total
    
def evalModelU(args, net, dataset, test_data, coords, num_chemical, writer, step, finalFlag=False, varlist=None, save_raw=False, raw_out_dir=None):
    device = args.device
    net.eval()
    steps = test_data.shape[1] - 1
    num_channels = test_data.shape[2]

    num_chemical = dataset["num_chemical"]
    num_temperature = dataset["num_temperature"]
    num_density = dataset["num_density"]
    num_velocity = dataset["num_velocity"]
    num_pressure = dataset["num_pressure"]

    with torch.no_grad():
        test_data = test_data.to(device)
        total_losses = dict(chemical=0.0, temperature=0.0, density=0.0, velocity=0.0, pressure=0.0)
        MeanPSNR = 0.0
        MeanSSIM = 0.0
        MeanMSSSIM = 0.0
        MeanGMSD = 0.0
        MeanMSGMSD = 0.0

        inference_time_total = 0.0
        for t in range(steps):
            inference_start = time.time()
            if t == 0:
                input_data = net.encoder(test_data[:, 0])

            output_pred = net(input_data, coords, None)
            output_gth = net.encoder(test_data[:, t + 1])

            input_data = output_pred

            inference_end = time.time()
            inference_time = inference_end - inference_start
            inference_time_total += inference_time

            diff = output_pred - output_gth
            total_losses['chemical'] += (diff[:, :num_chemical] ** 2).mean().item()
            total_losses['temperature'] += (diff[:, num_chemical:num_chemical + num_temperature] ** 2).mean().item()
            if num_density > 0:
                total_losses['density'] += (diff[:, num_chemical + num_temperature:num_chemical + num_temperature + num_density] ** 2).mean().item()
            total_losses['velocity'] += (diff[:, num_chemical + num_temperature + num_density:num_chemical + num_temperature + num_density + num_velocity] ** 2).mean().item()
            if num_pressure > 0:
                total_losses['pressure'] += (diff[:, num_chemical + num_temperature + num_density + num_velocity:] ** 2).mean().item()
                
            if finalFlag and save_raw and raw_out_dir is not None:
                pred_denorm = net.decoder(output_pred.clone())   # [B,C,H,W]
                gt_denorm   = net.decoder(output_gth.clone())    # [B,C,H,W]

                B = pred_denorm.shape[0]
                raw_coords_np = np.array(dataset["coords"])  # 原始坐标（保持原样）

                for iCase in range(B):
                    pred_np = pred_denorm[iCase].detach().cpu().numpy()  # [C,H,W]
                    gt_np   = gt_denorm[iCase].detach().cpu().numpy()    # [C,H,W]

                    save_path = os.path.join(raw_out_dir, f"case{iCase}_t{t+1:04d}.npz")
                    np.savez_compressed(
                        save_path,
                        pred=pred_np,
                        gt=gt_np,
                        vars=np.array(varlist, dtype=object),
                        coords=raw_coords_np,
                        time_index=t+1
                    )
                    
            if finalFlag:
                for iCase in range(test_data.shape[0]):
                        
                    # Get Metric
                    PSNR = 0.0
                    SSIM = 0.0
                    MSSSIM = 0.0
                    GMSD = 0.0
                    MSGMSD = 0.0
                    for iC in range(num_channels):
                        pred_data = output_pred[iCase, iC, :].detach().cpu().numpy()
                        gt_data = output_gth[iCase, iC, :].detach().cpu().numpy()
                        x_coords = coords[0, 0, :].detach().cpu().numpy()
                        y_coords = coords[0, 1, :].detach().cpu().numpy()
                        vmin = min(gt_data.min(), pred_data.min())
                        vmax = max(gt_data.max(), pred_data.max())

                        fig_width = 8
                        fig_height = 1

                        gt_img_array = create_grayscale_scatter(x_coords, y_coords, gt_data, 
                            vmin=vmin, vmax=vmax, figsize=(fig_width, fig_height))
                        pred_img_array = create_grayscale_scatter(x_coords, y_coords, pred_data, 
                            vmin=vmin, vmax=vmax, figsize=(fig_width, fig_height))
                        
                        colored_pred_img, colored_gth_img = apply_colormap(pred_img_array, gt_img_array, cmap='jet')
                        
                        writer.add_image(f"Traj{iCase}_{varlist[iC][:-4]}_image/{t+1}_GT", colored_gth_img, global_step=0)
                        writer.add_image(f"Traj{iCase}_{varlist[iC][:-4]}_image/{t+1}_Pred", colored_pred_img, global_step=0)

                        pred_img = torch.from_numpy(pred_img_array).float().unsqueeze(0) / 255.0
                        gth_img = torch.from_numpy(gt_img_array).float().unsqueeze(0) / 255.0
                        pred_img = pred_img.unsqueeze(0)
                        gth_img = gth_img.unsqueeze(0)

                        PSNR += calculate_psnr(gth_img, pred_img)
                        SSIM += calculate_ssim(gth_img, pred_img)
                        MSSSIM += calculate_ms_ssim(gth_img, pred_img)
                        GMSD += calculate_gmsd(gth_img, pred_img)
                        MSGMSD += calculate_ms_gmsd(gth_img, pred_img)

                    PSNR /= output_gth.shape[1]
                    SSIM /= output_gth.shape[1]
                    MSSSIM /= output_gth.shape[1]
                    GMSD /= output_gth.shape[1]
                    MSGMSD /= output_gth.shape[1]

                    MeanPSNR += PSNR
                    MeanSSIM += SSIM
                    MeanMSSSIM += MSSSIM
                    MeanGMSD += GMSD
                    MeanMSGMSD += MSGMSD
        
        MeanPSNR /= steps * test_data.shape[0]
        MeanSSIM /= steps * test_data.shape[0]
        MeanMSSSIM /= steps * test_data.shape[0]
        MeanGMSD /= steps * test_data.shape[0]
        MeanMSGMSD /= steps * test_data.shape[0]

        total_loss = sum(total_losses.values())
        for k, v in total_losses.items():
            print(f"  {k.capitalize():<12}: {v:.6f}")
        print(f"  TOTAL        : {total_loss:.6f}")
        return total_loss, MeanPSNR, MeanSSIM, MeanMSSSIM, MeanGMSD, MeanMSGMSD, inference_time_total
    
def evalModelGraph(args, net, dataset, test_data, coords, num_chemical, writer, step, finalFlag=False, varlist=None, save_raw=False, raw_out_dir=None):
    device = args.device
    net.eval()
    steps = test_data.shape[1] - 1
    num_channels = test_data.shape[2]
    
    test_data = test_data.to(device)
    test_dataset = ODEGraphDatasetRollout(test_data, coords)
    test_loader = DataLoader(test_dataset, batch_size=1)

    num_chemical = dataset["num_chemical"]
    num_temperature = dataset["num_temperature"]
    num_density = dataset["num_density"]
    num_velocity = dataset["num_velocity"]
    num_pressure = dataset["num_pressure"]

    with torch.no_grad():
        total_losses = dict(chemical=0.0, temperature=0.0, density=0.0, velocity=0.0, pressure=0.0)
        MeanPSNR = 0.0
        MeanSSIM = 0.0
        MeanMSSSIM = 0.0
        MeanGMSD = 0.0
        MeanMSGMSD = 0.0
        
        inference_time_total = 0.0
        for b_i, graph_batch in enumerate(test_loader):
            y = graph_batch.y
            steps = y.shape[1]
            for t in range(steps):
                inference_start = time.time()
                if t == 0:
                    graph_batch.input_fields = net.encoder(graph_batch.input_fields)
                output_pred = net.model(graph_batch, roll_out=True)
                output_gth = net.encoder(y[:, t])
                
                graph_batch.input_fields = output_pred
                
                inference_end = time.time()
                inference_time = inference_end - inference_start
                inference_time_total += inference_time
                            
                total_losses['chemical'] += ((output_pred - output_gth)[:, :num_chemical] ** 2).mean().item()
                total_losses['temperature'] += ((output_pred - output_gth)[:, num_chemical:num_chemical + num_temperature] ** 2).mean().item()
                total_losses['density'] += ((output_pred - output_gth)[:, num_chemical + num_temperature:num_chemical + num_temperature + num_density] ** 2).mean().item()
                total_losses['velocity'] += ((output_pred - output_gth)[:, num_chemical + num_temperature + num_density:] ** 2).mean().item()
                
                if finalFlag and save_raw and raw_out_dir is not None:
                    pred_denorm = net.decoder(output_pred.clone())   # [B,C,H,W]
                    gt_denorm   = net.decoder(output_gth.clone())    # [B,C,H,W]

                    B = 1
                    pred_denorm = pred_denorm.view(B, -1, output_pred.shape[-1]).permute(0, 2, 1)
                    gt_denorm = gt_denorm.view(B, -1, output_pred.shape[-1]).permute(0, 2, 1)

                    raw_coords_np = np.array(dataset["coords"])  # 原始坐标（保持原样）

                    pred_np = pred_denorm[0].detach().cpu().numpy()  # [C,H,W]
                    gt_np   = gt_denorm[0].detach().cpu().numpy()    # [C,H,W]

                    save_path = os.path.join(raw_out_dir, f"case{b_i}_t{t+1:04d}.npz")
                    np.savez_compressed(
                        save_path,
                        pred=pred_np,
                        gt=gt_np,
                        vars=np.array(varlist, dtype=object),
                        coords=raw_coords_np,
                        time_index=t+1
                    )
                    
                if finalFlag:
                    # Get Metric
                    PSNR = 0.0
                    SSIM = 0.0
                    MSSSIM = 0.0
                    GMSD = 0.0
                    MSGMSD = 0.0
                    for iC in range(num_channels):
                        pred_data = output_pred[:, iC].detach().cpu().numpy()
                        gt_data = output_gth[:, iC].detach().cpu().numpy()
                        x_coords = coords[0, 0, :].detach().cpu().numpy()
                        y_coords = coords[0, 1, :].detach().cpu().numpy()
                        vmin = min(gt_data.min(), pred_data.min())
                        vmax = max(gt_data.max(), pred_data.max())

                        fig_width = 8
                        fig_height = 1

                        gt_img_array = create_grayscale_scatter(x_coords, y_coords, gt_data, 
                            vmin=vmin, vmax=vmax, figsize=(fig_width, fig_height))
                        pred_img_array = create_grayscale_scatter(x_coords, y_coords, pred_data, 
                            vmin=vmin, vmax=vmax, figsize=(fig_width, fig_height))
                        
                        colored_pred_img, colored_gth_img = apply_colormap(pred_img_array, gt_img_array, cmap='jet')
                        
                        writer.add_image(f"Traj{b_i}_{varlist[iC][:-4]}_image/{t+1}_GT", colored_gth_img, global_step=0)
                        writer.add_image(f"Traj{b_i}_{varlist[iC][:-4]}_image/{t+1}_Pred", colored_pred_img, global_step=0)

                        pred_img = torch.from_numpy(pred_img_array).float().unsqueeze(0) / 255.0
                        gth_img = torch.from_numpy(gt_img_array).float().unsqueeze(0) / 255.0
                        pred_img = pred_img.unsqueeze(0)
                        gth_img = gth_img.unsqueeze(0)

                        PSNR += calculate_psnr(gth_img, pred_img)
                        SSIM += calculate_ssim(gth_img, pred_img)
                        MSSSIM += calculate_ms_ssim(gth_img, pred_img)
                        GMSD += calculate_gmsd(gth_img, pred_img)
                        MSGMSD += calculate_ms_gmsd(gth_img, pred_img)

                    PSNR /= output_gth.shape[1]
                    SSIM /= output_gth.shape[1]
                    MSSSIM /= output_gth.shape[1]
                    GMSD /= output_gth.shape[1]
                    MSGMSD /= output_gth.shape[1]

                    MeanPSNR += PSNR
                    MeanSSIM += SSIM
                    MeanMSSSIM += MSSSIM
                    MeanGMSD += GMSD
                    MeanMSGMSD += MSGMSD
        
        MeanPSNR /= steps * test_data.shape[0]
        MeanSSIM /= steps * test_data.shape[0]
        MeanMSSSIM /= steps * test_data.shape[0]
        MeanGMSD /= steps * test_data.shape[0]
        MeanMSGMSD /= steps * test_data.shape[0]

        total_loss = sum(total_losses.values())
        for k, v in total_losses.items():
            print(f"  {k.capitalize():<12}: {v:.6f}")
        print(f"  TOTAL        : {total_loss:.6f}")
        return total_loss, MeanPSNR, MeanSSIM, MeanMSSSIM, MeanGMSD, MeanMSGMSD, inference_time_total