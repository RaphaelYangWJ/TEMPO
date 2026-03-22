import torch
import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cox_ph_loss_static(risk_scores, durations, events):

    loss = 0
    event_indices = torch.where(events == 1)[0]

    if len(event_indices) == 0:
        return torch.tensor(0.0, requires_grad=True)

    for i in event_indices:
        risk_set_mask = (durations >= durations[i])
        log_sum_exp = torch.logsumexp(risk_scores[risk_set_mask], dim=0)
        loss += (risk_scores[i] - log_sum_exp)
    return -loss / len(event_indices)