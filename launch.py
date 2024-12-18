from import_data import load_data_from_directory
import torch
import lightning
from model import Model
import random

def step(self, batch, batch_idx) :
        loss=torch.tensor(0., requires_grad = True)
        for graph in batch :
            loss = loss + self.model.loss(graph)
        
        loss=loss/len(batch)

        return loss

class LitAutoEncoder(lightning.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model=model
    
    def training_step(self, batch, batch_idx) :
        train_loss=step(self,batch,batch_idx)
        self.log("train_loss_step", train_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=len(batch))
        return train_loss

    def validation_step(self, batch, batch_idx) :
        valid_loss=step(self,batch,batch_idx)
        self.log("val_loss_step", valid_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, batch_size=len(batch))
        return valid_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

torch.set_float32_matmul_precision('high')

graphs=load_data_from_directory()
print("Number of graphs : ",len(graphs))
random.shuffle(graphs)

train_graphs=graphs[:int(len(graphs)*0.9)]
valid_graphs=graphs[int(len(graphs)*0.9):]

model=Model(hidden_size=10)

autoencoder=LitAutoEncoder(model)

def collate_fn(data) :
    return data
train_dataloader=torch.utils.data.DataLoader(train_graphs,batch_size=50,shuffle=True,collate_fn=collate_fn,num_workers=31)
valid_dataloader=torch.utils.data.DataLoader(valid_graphs,batch_size=50,collate_fn=collate_fn,num_workers=31)

checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
    dirpath="checkpoints/",
    filename="checkpoint-{epoch:02d}",
    save_top_k=-1
)

trainer = lightning.Trainer(
        max_epochs=200,
        accelerator="cpu",
        num_sanity_val_steps=0,
        logger=lightning.pytorch.loggers.CSVLogger("logs/", name="log"),
        log_every_n_steps=1,
        callbacks=[checkpoint_callback]
    )

# Training
trainer.fit(model=autoencoder, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

# Uncomment the following line to load a checkpoint
#autoencoder=LitAutoEncoder.load_from_checkpoint("checkpoints/checkpoint-epoch=XX.ckpt",model=model)

# The following function will return a sample
#autoencoder.model.sample()