from voxcpm import VoxCPM2

model = VoxCPM2.from_pretrained("openbmb/VoxCPM2")

# Voice design mode
audio = model.synthesize(
    text="The Punters' Club; good music, tolerable hosts, and disco shoulders",
    voice_description="warm, calm, female, mid-30s, UK, London accent"
)