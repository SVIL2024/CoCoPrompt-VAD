import torch
import torch.nn as nn


# =============================================================================
# Original Loss Functions
# =============================================================================

def sparsity(arr, lambda_sparse):
    loss = torch.mean(arr)
    return lambda_sparse * loss


def smooth(arr, lambda_smooth):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2 - arr) ** 2)

    return lambda_smooth * loss


def inter_class_margin_loss(similarity_topk, labels_topk_adjusted, margin=0.3, focal_gamma=0.0):
    """Logit-based margin loss with optional focal weighting for hard samples."""
    num_classes = similarity_topk.shape[1]

    labels_clamped = labels_topk_adjusted.clamp(0, num_classes - 1)

    correct_logits = similarity_topk.gather(1, labels_clamped.unsqueeze(1)).squeeze(1)

    mask = torch.ones_like(similarity_topk, dtype=torch.bool)
    mask.scatter_(1, labels_clamped.unsqueeze(1), False)

    other_max_logits = similarity_topk.masked_fill(~mask, float('-inf')).max(dim=1)[0]

    raw_loss = torch.clamp(margin + other_max_logits - correct_logits, min=0)
    
    if focal_gamma > 0:
        probs = torch.softmax(similarity_topk, dim=1)
        correct_probs = probs.gather(1, labels_clamped.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - correct_probs) ** focal_gamma
        loss = (raw_loss * focal_weight).mean()
    else:
        loss = raw_loss.mean()

    return loss


def direction_orthogonality_loss(text_directions):
    """Penalize off-diagonal cosine similarities between class direction vectors.

    Args:
        text_directions: (num_classes, embed_dim) L2-normalized direction vectors
    """
    cos_sim = text_directions @ text_directions.T
    mask = 1 - torch.eye(cos_sim.shape[0], device=cos_sim.device)
    loss = (cos_sim * mask).pow(2).sum() / (mask.sum() + 1e-8)
    return loss


def temporal_class_smoothness_loss(softmax_sim, scores, num_segments, frames_per_segment):
    """Temporal smoothness on class probabilities within each video (no cross-boundary).
    
    Args:
        softmax_sim: (batch*num_segments*frames_per_segment, num_classes-1) softmax of similarity
        scores: (batch*num_segments*frames_per_segment,) anomaly scores
        num_segments: number of segments per video
        frames_per_segment: frames per segment
    """
    batch_size = softmax_sim.shape[0] // (num_segments * frames_per_segment)
    num_classes = softmax_sim.shape[1]
    
    sim_reshaped = softmax_sim.view(batch_size, num_segments * frames_per_segment, num_classes)
    scores_reshaped = scores.view(batch_size, num_segments * frames_per_segment)
    
    sim_t = sim_reshaped[:, :-1, :]
    sim_t1 = sim_reshaped[:, 1:, :]
    scores_t = scores_reshaped[:, :-1]
    scores_t1 = scores_reshaped[:, 1:]
    
    weight = torch.clamp(torch.minimum(scores_t, scores_t1), min=0, max=1)
    
    diff = ((sim_t1 - sim_t) ** 2).sum(dim=-1)
    loss = (diff * weight).mean()
    
    return loss


class ComputeLoss:
    def __init__(
        self,
        normal_id,
        num_topk,
        lambda_dir_abn,
        lambda_dir_nor,
        lambda_topk_abn,
        lambda_bottomk_abn,
        lambda_topk_nor,
        lambda_smooth,
        lambda_sparse,
        frames_per_segment,
        num_segments,
        lambda_margin=0.0,
        margin=0.3,
        lambda_dor=0.0,
        lambda_tcs=0.0,
        focal_gamma=2.0,
        class_weights=None,
    ):
        super().__init__()

        self.normal_id = normal_id
        self.num_topk = num_topk
        self.lambda_dir_abn = lambda_dir_abn
        self.lambda_dir_nor = lambda_dir_nor
        self.lambda_topk_abn = lambda_topk_abn
        self.lambda_bottomk_abn = lambda_bottomk_abn
        self.lambda_topk_nor = lambda_topk_nor
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse
        self.frames_per_segment = frames_per_segment
        self.num_segments = num_segments
        self.lambda_margin = lambda_margin
        self.margin = margin
        self.lambda_dor = lambda_dor
        self.lambda_tcs = lambda_tcs
        self.focal_gamma = focal_gamma
        self.class_weights = class_weights

        self.loss_bce = torch.nn.BCELoss()

    def __call__(
        self,
        similarity,
        similarity_topk,
        labels,
        scores,
        idx_topk_abn,
        idx_topk_nor,
        idx_bottomk_abn,
        text_directions=None,
    ):
        # print("similarity.shape", similarity.shape) # 64*32*16, 13
        # print("similarity_topk.shape", similarity_topk.shape) # 64*k_abn*16, 13
        # print("scores.shape", scores.shape) # 64*32*16, 1

        alabels = labels[: labels.shape[0] // 2]
        nlabels = labels[labels.shape[0] // 2 :]

        repeat_interleave_params = [
            (alabels, self.num_segments * self.frames_per_segment),
            (alabels, self.num_topk * self.frames_per_segment),
            (nlabels, self.num_topk * self.frames_per_segment),
        ]

        alabels_per_frame, alabels_per_topk, nlabels_per_topk = (
            label.repeat_interleave(repeat) for label, repeat in repeat_interleave_params
        )

        asimilarity_topk = similarity_topk[: alabels_per_topk.shape[0]]

        nsimilarity = similarity[similarity.shape[0] // 2 :]

        alabels_per_frame[alabels_per_frame > self.normal_id] -= 1
        alabels_per_topk[alabels_per_topk > self.normal_id] -= 1

        asimilarity_topk_total = []
        asimilarity_topk_weights = []

        # Class reweighting for rare classes (XD-Violence: Abuse=0, Shooting=5 after normal_id adjustment)
        # Default weight=1.0, rare classes get higher weights
        class_weights = getattr(self, 'class_weights', None)

        for c in range(similarity.shape[1]):
            # Loss on abnormal ones
            abnormal_indices_topk = (alabels_per_topk == c).nonzero(as_tuple=True)[0]

            if abnormal_indices_topk.nelement():
                asimilarity_topk_c = asimilarity_topk[abnormal_indices_topk, c]  # (bs_c*k*seg_len)

                asimilarity_topk_total.append(asimilarity_topk_c)
                
                # Apply class-specific weight
                if class_weights is not None and c < len(class_weights):
                    weight = class_weights[c]
                else:
                    weight = 1.0
                asimilarity_topk_weights.extend([weight] * asimilarity_topk_c.numel())

        asimilarity_topk_total = torch.cat(asimilarity_topk_total, dim=0)

        # on most abnormal - with optional class reweighting
        if class_weights is not None and len(asimilarity_topk_weights) > 0:
            weights_tensor = torch.tensor(asimilarity_topk_weights, device=asimilarity_topk_total.device)
            ldir_abn = self.lambda_dir_abn * -1.0 * (asimilarity_topk_total * weights_tensor).sum() / weights_tensor.sum()
        else:
            ldir_abn = self.lambda_dir_abn * -1.0 * asimilarity_topk_total.mean(dim=0)

        # on all normal
        ldir_nor = (nsimilarity.max(dim=1)[0]).mean(dim=0)
        ldir_nor = self.lambda_dir_nor * ldir_nor

        num_classes = similarity.shape[1] + 1  # +1 for normal class
        # compute probability for each frame to be of each class given the abnormality score
        softmax_similarity = torch.softmax(similarity, dim=1)
        # element-wise multiplication
        class_probs = softmax_similarity * scores.unsqueeze(1)  # shape: [1024, 1]
        # add normal probability to the class probabilities
        normal_probs = 1 - scores
        normal_probs = normal_probs.unsqueeze(1)  # Add a new dimension to match class_probs shape
        class_probs = torch.cat(
            (
                class_probs[:, : self.normal_id],
                normal_probs,
                class_probs[:, self.normal_id :],
            ),
            dim=1,
        )
        class_probs = class_probs.view(-1, self.num_segments, self.frames_per_segment, num_classes)
        aclass_probs = class_probs[: class_probs.shape[0] // 2]
        nclass_probs = class_probs[class_probs.shape[0] // 2 :]

        aclass_probs_topk = torch.gather(
            aclass_probs,
            1,
            idx_topk_abn.unsqueeze(2)
            .unsqueeze(3)
            .expand([-1, -1, self.frames_per_segment, num_classes]),
        )
        aclass_probs_bottomk = torch.gather(
            aclass_probs,
            1,
            idx_bottomk_abn.unsqueeze(2)
            .unsqueeze(3)
            .expand([-1, -1, self.frames_per_segment, num_classes]),
        )

        aclass_probs_topk = aclass_probs_topk.view(-1, num_classes)
        aclass_probs_bottomk = aclass_probs_bottomk.view(-1, num_classes)

        loss_fn = nn.NLLLoss()

        # Compute the log probabilities
        aclass_log_probs_topk = torch.log(aclass_probs_topk)
        aclass_log_probs_bottomk = torch.log(aclass_probs_bottomk)

        # Save adjusted labels BEFORE restoring for NLLLoss (for margin loss)
        alabels_per_topk_adjusted = alabels_per_topk.clone()

        alabels_per_topk[alabels_per_topk >= self.normal_id] += 1
        # assert that normal id is not in abnormal labels
        assert torch.all(alabels_per_topk != self.normal_id), "Normal id is in abnormal labels"

        # Compute the loss
        ltopk_abn = loss_fn(aclass_log_probs_topk, alabels_per_topk)
        # create a tensor of the same size as aclass_probs_most_nor filled with the value of the normal class id
        lbottomk_abn = loss_fn(
            aclass_log_probs_bottomk,
            (torch.ones(aclass_log_probs_bottomk.shape[0]) * self.normal_id)
            .long()
            .to(similarity.device),
        )

        nclass_probs_topk = torch.gather(
            nclass_probs,
            1,
            idx_topk_nor.unsqueeze(2)
            .unsqueeze(3)
            .expand([-1, -1, self.frames_per_segment, num_classes]),
        )
        nclass_probs_topk = nclass_probs_topk.view(-1, num_classes)
        nclass_log_probs_topk = torch.log(nclass_probs_topk)
        assert torch.all(nlabels_per_topk == self.normal_id), "Not all normal labels are normal id"
        ltopk_nor = loss_fn(nclass_log_probs_topk, nlabels_per_topk)

        ltopk_abn = self.lambda_topk_abn * ltopk_abn
        lbottomk_abn = self.lambda_bottomk_abn * lbottomk_abn
        ltopk_nor = self.lambda_topk_nor * ltopk_nor

        # Smooth & Sparsity terms
        abn_scores = scores[: scores.shape[0] // 2]
        lsmooth = smooth(abn_scores, self.lambda_smooth)
        lsparse = sparsity(abn_scores, self.lambda_sparse)

        if self.lambda_margin > 0:
            lmargin = self.lambda_margin * inter_class_margin_loss(
                asimilarity_topk,
                alabels_per_topk_adjusted,
                margin=self.margin,
                focal_gamma=self.focal_gamma,
            )
        else:
            lmargin = torch.tensor(0.0, device=similarity.device)

        if self.lambda_dor > 0 and text_directions is not None:
            ldor = self.lambda_dor * direction_orthogonality_loss(text_directions)
        else:
            ldor = torch.tensor(0.0, device=similarity.device)

        if self.lambda_tcs > 0:
            ltcs = self.lambda_tcs * temporal_class_smoothness_loss(
                softmax_similarity, scores, self.num_segments, self.frames_per_segment
            )
        else:
            ltcs = torch.tensor(0.0, device=similarity.device)

        cost = ldir_abn + ldir_nor + ltopk_abn + lbottomk_abn + ltopk_nor + lsmooth + lsparse + lmargin + ldor + ltcs

        return (
            cost,
            ldir_abn,
            ldir_nor,
            ltopk_abn,
            lbottomk_abn,
            ltopk_nor,
            lsmooth,
            lsparse,
        )
