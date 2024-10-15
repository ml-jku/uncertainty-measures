import torch

def calculate_uncertainties(
    pred_probs: torch.Tensor,
    comp_probs: torch.Tensor,
    eps: float = 1e-10    
):
    """
    Calculate all proposed measures of uncertainty.
    Treats the last dimension as class probabilities, make sure they sum up to 1.
    Averages (for posterior expectations) are calculated over the second to last dimension.
    Measures that use a single model predicting model use the first model in the second to last dimension.
    Measures approximating the true model with a single model use the last model in the second to last dimension.

    Input: 
    - pred_probs: [..., n_models, n_classes]
    - comp_probs: [..., n_models, n_classes]
    - eps: small value for numerical stability
    Output:
    - dict with all uncertainties
    """

    same_samples = (pred_probs.shape == comp_probs.shape) and torch.allclose(pred_probs, comp_probs, atol=1e-5)

    # get posterior predictives
    avg_pred_probs = torch.mean(pred_probs, dim=-2)
    avg_comp_probs = torch.mean(comp_probs, dim=-2)

    # get first and last model in the second to last dimension
    single_pred_probs = pred_probs[..., 0, :]  
    single_comp_probs = comp_probs[..., -1, :]

    # calculate total uncertainties
    total_single_single = - torch.sum(single_pred_probs * torch.log(single_comp_probs + eps), dim=-1)
    total_single_avg = - torch.sum(single_pred_probs * torch.log(avg_comp_probs + eps), dim=-1)
    total_single_exp = - torch.mean(torch.sum(single_pred_probs.unsqueeze(-2) * torch.log(comp_probs + eps), dim=-1), dim=-1)
    total_avg_single = - torch.sum(avg_pred_probs * torch.log(single_comp_probs + eps), dim=-1)
    total_avg_avg = - torch.sum(avg_pred_probs * torch.log(avg_comp_probs + eps), dim=-1)
    total_avg_exp = - torch.mean(torch.sum(avg_pred_probs.unsqueeze(-2) * torch.log(comp_probs + eps), dim=-1), dim=-1)
    total_exp_single = - torch.mean(torch.sum(pred_probs * torch.log(single_comp_probs.unsqueeze(-2) + eps), dim=-1), dim=-1)
    total_exp_avg = - torch.mean(torch.sum(pred_probs * torch.log(avg_comp_probs.unsqueeze(-2) + eps), dim=-1), dim=-1)

    # calculate aleatoric uncertainties
    ale_single = - torch.sum(single_pred_probs * torch.log(single_pred_probs + eps), dim=-1)
    ale_avg = - torch.sum(avg_pred_probs * torch.log(avg_pred_probs + eps), dim=-1)
    ale_exp = - torch.mean(torch.sum(pred_probs * torch.log(pred_probs + eps), dim=-1), dim=-1)

    # calculate epistemic uncertainties by subtracting total and aleatoric uncertainties
    # relu is used to avoid negative uncertainties -> caused by numerical instability of subtracting two very similar numbers
    epi_single_single = torch.relu(total_single_single - ale_single)
    epi_single_avg = torch.relu(total_single_avg - ale_single)
    epi_single_exp = torch.relu(total_single_exp - ale_single)
    epi_avg_single = torch.relu(total_avg_single - ale_avg)
    epi_avg_avg = torch.relu(total_avg_avg - ale_avg)
    epi_avg_exp = torch.relu(total_avg_exp - ale_avg)
    epi_exp_single = torch.relu(total_exp_single - ale_exp)
    epi_exp_avg = torch.relu(total_exp_avg - ale_exp)
    
    if same_samples:
        # double expectation can be calculated by adding the two single expectations
        # follows from jensen's inequality
        epi_exp_exp = epi_exp_avg + epi_avg_exp
        total_exp_exp = ale_exp + epi_exp_exp
    else:
        total_exp_exp = - torch.mean(torch.mean(
            torch.sum(pred_probs.unsqueeze(-3) * torch.log(comp_probs.unsqueeze(-2) + eps), dim=-1), 
            dim=-1), dim=-1)
        epi_exp_exp = torch.relu(total_exp_exp - ale_exp)


    return {
        "A1": [total_single_single, ale_single, epi_single_single],
        "A2": [total_single_avg, ale_single, epi_single_avg],
        "A3": [total_single_exp, ale_single, epi_single_exp],
        "B1": [total_avg_single, ale_avg, epi_avg_single],
        "B2": [total_avg_avg, ale_avg, epi_avg_avg],
        "B3": [total_avg_exp, ale_avg, epi_avg_exp],
        "C1": [total_exp_single, ale_exp, epi_exp_single],
        "C2": [total_exp_avg, ale_exp, epi_exp_avg],
        "C3": [total_exp_exp, ale_exp, epi_exp_exp]
    }
