from PSPNet.builder import PSPNet

psp = PSPNet(
    blocks=1,
    start_filters=12
)

psp.net.summary()
