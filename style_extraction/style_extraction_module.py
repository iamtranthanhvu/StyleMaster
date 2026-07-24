import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from controlnet_aux import ContentShuffleDetector
from basic_modules import Transformer
import torch.nn as nn
import torch.nn.functional as F

class Processor(nn.Module):
    def __init__(
            self,
            image_encoder_ckpt_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
            ) -> None:
        super(Processor, self).__init__()

        self.image_encoder = self.init_image_encoder(image_encoder_ckpt_path)
        self.init_other_components()
    
    def init_image_encoder(self, image_encoder_ckpt_path):
        """Initialize the image encoder."""
        return CLIPVisionModelWithProjection.from_pretrained(image_encoder_ckpt_path)
    
    def init_other_components(self):
        """Initialize other components."""
        self.clip_image_processor = CLIPImageProcessor()
        self.content_shuffler = ContentShuffleDetector()
    
    def process_images(self, ref_images):
        ref_images = [image.convert('RGB') for image in ref_images]
        ps_images = [self.content_shuffler(x) for x in ref_images]
        try:
            clip_image = self.clip_image_processor(images=ref_images, return_tensors="pt").pixel_values
            # do content shuffle
            clip_image_ps = self.clip_image_processor(images=ps_images, return_tensors="pt").pixel_values
            
            image_embeds = self.image_encoder(clip_image.to(dtype=self.image_encoder.dtype,device=self.image_encoder.device))
            image_embeds_ps = self.image_encoder(clip_image_ps.to(dtype=self.image_encoder.dtype,device=self.image_encoder.device))
                
            self.image_embeds_ps = image_embeds_ps
            self.image_embeds = image_embeds
                
        
        except Exception as e:
            print(f"An error occurred: {e}")
            image_embeds = self.image_embeds
            image_embeds_ps = self.image_embeds_ps
            image_embeds_ps.image_embeds = torch.zeros_like(self.image_embeds_ps.image_embeds)
            image_embeds.last_hidden_state = torch.zeros_like(self.image_embeds.last_hidden_state)
        
        return image_embeds, image_embeds_ps


class StyleModel(nn.Module):
    def __init__(
        self,
        local_tokens=2,
        n_layers=3,
        num_heads=8,
        in_dim=1280,
        clip_embeddings_dim=1024,
        global_tokens=14,
        cross_attention_dim=4096,
        proj_model_ckpt_path='./checkpoints/image_proj_model.ckpt',
    ):
        super(StyleModel, self).__init__()
        

        # Initialize Projection Model
        self.image_proj_model = self.init_projection_model(
            local_tokens, n_layers, num_heads, in_dim, 
            clip_embeddings_dim, global_tokens, cross_attention_dim, 
            proj_model_ckpt_path
        )

    def init_projection_model(
        self, local_tokens, n_layers, num_heads, in_dim, 
        clip_embeddings_dim, global_tokens, cross_attention_dim, 
        proj_model_ckpt_path
    ):
        """Initialize the projection model."""
        # in_dim: the dimension of clip output (image_embeds.hidden_states:1280)
        self.in_dim = in_dim
        # clip_embeddings_dim: (image_embeds.image_embeds:1024)
        self.clip_embeddings_dim = clip_embeddings_dim
        # output number of style tokens (local + global)
        self.local_tokens = local_tokens
        # setting of q-former transformer layer
        self.n_layers = n_layers
        self.num_heads = num_heads
        # output dimension of style token
        self.cross_attention_dim = cross_attention_dim

        scale = self.in_dim ** -0.5
        self.global_tokens = global_tokens
        self.local_tokens = local_tokens
        # q-former learnable tokens
        self.style_emb = nn.Parameter(torch.randn(1, self.local_tokens, self.in_dim) * scale)
        self.transformer_blocks = Transformer(
            width=self.in_dim,
            layers=self.n_layers,
            heads=self.num_heads,
        )
        self.ln1 = nn.LayerNorm(self.in_dim)
        self.ln2 = nn.LayerNorm(self.in_dim)
        # local token projection
        
        self.proj_patch = nn.Parameter(torch.randn(self.in_dim, self.cross_attention_dim) * scale)
        #global tokens
        self.global_tokens = global_tokens
        self.proj = nn.Linear(self.clip_embeddings_dim, self.global_tokens * self.cross_attention_dim)
        self.norm = nn.LayerNorm(self.cross_attention_dim)

        # Load pretrained projection model
        if proj_model_ckpt_path is not None:
            print(f'load style model from {proj_model_ckpt_path}')
            self.load_proj_model(proj_model_ckpt_path)

    def load_proj_model(self, ckpt_path):
        """Load the pretrained projection model."""
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.load_state_dict(checkpoint['state_dict'],strict=True)
            else:
                self.load_state_dict(checkpoint)
            print(f"Successfully loaded projection model checkpoint from {ckpt_path}")
        except Exception as e:
            print(f"Error loading projection model checkpoint: {e}")

    def compress_prompt(self, prompt):
        prompt_compressed = F.adaptive_avg_pool1d(prompt, 1280)
        return prompt_compressed

    def drop_tokens_by_similarity(self, patch_tokens, prompt, drop_rate=0.5):
        # align prompt dimension with style token dimension
        prompt_compressed = self.compress_prompt(prompt)
        batch_size, sequence_length, feature_dim = patch_tokens.size()
        # calculate drop tokens
        num_to_drop = int(sequence_length * drop_rate)
        prompt_avg_token = prompt_compressed.mean(dim=1, keepdim=True)
        patch_tokens_norm = F.normalize(patch_tokens, dim=2)
        prompt_avg_token_norm = F.normalize(prompt_avg_token, dim=2).to(dtype=patch_tokens_norm.dtype)
        # compute similarity
        similarity = torch.bmm(patch_tokens_norm, prompt_avg_token_norm.permute(0, 2, 1)).squeeze(2)
        indices = similarity.argsort(dim=1)
        drop_indices = indices[:, :num_to_drop]
        mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=patch_tokens.device)
        mask.scatter_(1, drop_indices, True)
        patch_tokens_kept = patch_tokens[~mask].view(batch_size, sequence_length - num_to_drop, feature_dim)
        return patch_tokens_kept
    
    def project_image_embeddings(self, image_embeds, image_embeds_ps, prompt_embeds=None):
        # get patch features of image (local) 
        x = image_embeds.last_hidden_state

        # initialize q-former tokens for local features
        style_emb = self.style_emb.repeat(x.shape[0], 1, 1)
        
        # drop text-related patch features
        x = self.drop_tokens_by_similarity(x, prompt_embeds, 0.95)

        # concat to perform self-attention
        x = torch.cat([style_emb, x], dim=1)
        
        # q-former feature extraction of local feature
        x = self.ln1(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_blocks(x)
        x = x.permute(1, 0, 2)
        x = self.ln2(x[:, :self.local_tokens, :])
        x = x @ self.proj_patch

        # get global features of content shuffled image (global)
        x_ps = image_embeds_ps
        embeds = x_ps.image_embeds
        # use mlp to extract global features
        global_tokens = self.proj(embeds).reshape(
            -1, self.global_tokens, self.cross_attention_dim
        )
        global_tokens = self.norm(global_tokens)

        # merge local and global feature
        x = torch.cat([x, global_tokens], dim=1)
        return x

def main():
    # Initialize Processor
    processor = Processor()

    # Load sample images
    image_paths = [
        "assets/fine_1.jpg",
        "assets/800.jpeg"
    ]
    ref_images = []
    for path in image_paths:
        img = Image.open(path)
        ref_images.append(img)
    prompt_embeds = torch.randn(len(ref_images),77,4096)
    # Process images using Processor
    image_embeds, image_embeds_ps = processor.process_images(ref_images)

    # Initialize StyleModel
    style_model = StyleModel()

    # Project image embeddings using StyleModel
    projected_embeddings = style_model.project_image_embeddings(image_embeds, image_embeds_ps, prompt_embeds=prompt_embeds)

    # Print the shape of the projected embeddings
    print("Projected Embeddings Shape:", projected_embeddings.shape)

if __name__ == '__main__':
    main()