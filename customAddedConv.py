import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss

from transformers import ElectraModel
from transformers.models.electra.modeling_electra import ElectraPreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput


class customAddedConvModel(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config=config)
        
        self.electra = ElectraModel(config)
        self.conv_layer1  = ConvHead(config.hidden_size, out_channels=1024, kernel_size=1)
        self.conv_layer2  = ConvHead(config.hidden_size, out_channels=1024, kernel_size=3, padding=1)
        self.conv_layer3  = ConvHead(config.hidden_size, out_channels=1024, kernel_size=5, padding=2)
        self.drop_out  = nn.Dropout(0.3)
        self.classify_layer = nn.Linear(1024 * 3, 2, bias=True)

        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = discriminator_hidden_states[0]

        conv_output1 = self.conv_layer1(sequence_output)
        conv_output2 = self.conv_layer2(sequence_output)
        conv_output3 = self.conv_layer3(sequence_output)

        concat_output = torch.cat((conv_output1, conv_output2, conv_output3), dim=1) 

        concat_output = concat_output.transpose(1, 2) 
        concat_output = self.drop_out(concat_output) 
        logits = self.classify_layer(concat_output) 

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + discriminator_hidden_states[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


class ConvHead(nn.Module):
    def __init__(self, input_dim, out_channels=512, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.conv(x.transpose(1, 2)))
    