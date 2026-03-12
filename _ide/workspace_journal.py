# 2026-03-10T15:27:03.257177100
import vitis

client = vitis.create_client()
client.set_workspace(path="DP")

vitis.dispose()

