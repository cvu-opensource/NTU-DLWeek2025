from torch import nn
import torch
from llm2vec.models import LlamaBiModel
from info_nce import InfoNCE, info_nce

class DINOtext(nn.Module):
    """
    Extreme scuffed implementation of DINO. So barebones and minimal, I'm
    pretty sure it wont succeed. Oh well!
    Wrapper around LlamaBiModel (the llmtovec package)
    Creates seperate student and teacher self-distill architecture.

    """

    def __init__(self, 
                student_llama_model: LlamaBiModel,  # DIFFERENT INSTANCES FROM EACHOTHER
                teacher_llama_model: LlamaBiModel,  # DIFFERENT INSTANCES FROM EACHOTHER
                embed_dim, # TODO: use llamabimodel's
            ):
        #TODO document or not lol
        super().__init__()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        student = student_llama_model
        student.config.pad_token_id = student.config.eos_token_id
        student.to(device)
        teacher = teacher_llama_model
        teacher.config.pad_token_id = teacher.config.eos_token_id
        teacher.to(device)
        # atp backbones are done, wrap head
        temp_out_dim = 100
        student = MultiCropWrapper(student, DINOHead(
            embed_dim,
            temp_out_dim,
        ))
        teacher = MultiCropWrapper(
            teacher,
            DINOHead(embed_dim, temp_out_dim),
        )
        if has_batchnorms(self.student):
            student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
            teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

            # we need DDP wrapper to have synchro batch norms working...
            teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=0)
            teacher_without_ddp = teacher.module
        else:
            # teacher_without_ddp and teacher are the same thing
            teacher_without_ddp = teacher
    
        self.student = nn.parallel.DistributedDataParallel(student, device_ids=0)
        # teacher and student start with the same weights
        teacher_without_ddp.load_state_dict(self.student.module.state_dict())
        self.teacher = teacher
        # no grads for you papi
        for p in teacher.parameters():
            p.requires_grad = False
        print('Student teacher built lol')


    def forward(self, batch: dict):
        """
        batch is a dictionary of:
        {
            'global_crops_input_ids': torch.Tensor().size = N, C, D (C is crops not channels like in imgs)
            'global_crops_attention_mask': vice versa
            'local_crops_input_ids': vice versa
            'local_crops_attention_mask': vice versa
        }
        """
        global_crops_input_ids = batch['global_crops_input_ids']
        global_crops_attention_mask = batch['global_crops_attention_mask']
        self.student(input_ids=global_crops_input_ids, attention_mask=global_crops_attention_mask)
        global_crops = {} # we have N x global_crop samples. encode all at once
        for item in batch:
            for global_crop in item['global_crops']:
                global_crops.append(item['global_crops'])
        global_crops = torch.stack(global_crops, dim=1)
        print('global_crops input_ids shape:')
        teacher_output = self.teacher(images[:2])  # only the 2 global views pass through the teacher
        student_output = student(images)
        loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = self.momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(self.student.module.parameters(), self.teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

class DINOHead(nn.Module):
    def __init__(self,
                in_dim,
                out_dim,
                use_bn=False,
                norm_last_layer=True,
                nlayers=3,
                hidden_dim=2048,
                bottleneck_dim=256
            ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)


