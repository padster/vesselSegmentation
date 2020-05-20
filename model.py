PADDED_VOLUMES = {}
N_TRANSFORMS = 16

def transformParamsToID(flipXY, revX, revY, revZ):
  tID = 0
  tID = 2 * tID + (1 if flipXY else 0)
  tID = 2 * tID + (1 if   revX else 0)
  tID = 2 * tID + (1 if   revY else 0)
  tID = 2 * tID + (1 if   revZ else 0)
  return tID

def transformIDtoParams(tID):
  mask = int(tID)
  revZ,   mask = (mask & 1) != 0, mask >> 1
  revY,   mask = (mask & 1) != 0, mask >> 1
  revX,   mask = (mask & 1) != 0, mask >> 1
  flipXY, mask = (mask & 1) != 0, mask >> 1
  return flipXY, revX, revY, revZ


# ew.
def _HACK_SAFE_SLICE(idx):
	return None if idx == -1 else idx


def extractSubVolume(subvolume):
	assert subvolume.s in PADDED_VOLUMES
	volume = PADDED_VOLUMES[subvolume.s]
	pad = subvolume.p
	flipXY, revX, revY, revZ = transformIDtoParams(subvolume.t)
	x1, x2 = subvolume.x, subvolume.x + 2 * pad + 1
	y1, y2 = subvolume.y, subvolume.y + 2 * pad + 1
	z1, z2 = subvolume.z, subvolume.z + 2 * pad + 1
	sX = slice(x1, x2) if not revX else slice(x2 - 1, _HACK_SAFE_SLICE(x1 - 1), -1)
	sY = slice(y1, y2) if not revY else slice(y2 - 1, _HACK_SAFE_SLICE(y1 - 1), -1)
	sZ = slice(z1, z2) if not revZ else slice(z2 - 1, _HACK_SAFE_SLICE(z1 - 1), -1)
	unflipped = volume[sX, sY, sZ, :]
	if unflipped.shape[2] == 0:
		print ("HERE")
		print (z1, z2, volume.shape[2])
		print (sZ)
	return unflipped if not flipXY else unflipped.transpose(1, 0, 2, 3) # reverse X & Y

class SubVolume:
	s = None 						 # Scan ID, as a three-digit string
	x, y, z = -1, -1, -1 # Centre position
	p = -1               # Padding around centre
	t = -1               # Transform to apply

	def __init__(self, scanID, x, y, z, pad, flipXY, revX, revY, revZ):
		self.s = scanID
		self.x = x
		self.y = y
		self.z = z
		self.p = pad
		self.t = transformParamsToID(flipXY, revX, revY, revZ)

	def __repr__(self):
		fxy, rx, ry, rz = transformIDtoParams(self.t)
		return "Scan %s, xyz = (%d, %d, %d), pad = %d, flip xy = %s, reverse xyz = %s/%s/%s" % (self.s, self.x, self.y, self.z, self.p, fxy, rx, ry, rz)


def SubVolumeFromTransformID(scanID, x, y, z, pad, transformID):
	flipXY, revX, revY, revZ = transformIDtoParams(transformID)
	return SubVolume(scanID, x, y, z, pad, flipXY, revX, revY, revZ)
