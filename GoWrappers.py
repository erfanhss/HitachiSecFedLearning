from ctypes import *

# loading the shared libraries
lib = CDLL("./Encryption/func_ubuntu.so")

# defining the required conversion
def convertToGoSlice(npArray):
    data = (c_double * len(npArray))(0)
    for i in range(len(npArray)):
        data[i] = float(npArray[i])
    return GoSlice(data, len(data), len(data))

class GoString(Structure):
    _fields_ = [("p", c_char_p), ("n", c_longlong)]

class GoSlice(Structure):
    _fields_ = [("data", POINTER(c_double)), ("len", c_longlong), ("cap", c_longlong)]

class clientPhase1_return(Structure):
    _fields_ = [("r0", c_char_p), ("r1", c_char_p), ("r2", c_longlong)]

# defining the arguemetns of the imported functions
lib.clientPhase1.argtypes = [GoString, c_ubyte, c_ulonglong, c_double, c_double]
lib.clientPhase1.restype = clientPhase1_return
lib.serverPhase1.argtypes = [GoString, c_longlong, c_ubyte, c_ulonglong, c_double]
lib.clientPhase2.argtypes = [GoSlice, GoString, GoString, c_longlong, GoString, c_ubyte, c_ulonglong, c_double, c_double]
lib.serverPhase2.argtypes = [GoString, c_longlong, c_ubyte, c_double, c_ulonglong, c_double, c_longlong]
lib.serverPhase2.restype = c_char_p

# defining the wrapper functions
def client_phase1(server_address, robust, log_degree, log_scale, resiliency):
    out = lib.clientPhase1(GoString(server_address, len(server_address)), robust, log_degree, 2.**log_scale, resiliency)
    return GoString(out.r0, len(out.r0)), GoString(out.r1, len(out.r1)), out.r2

def server_phase1(server_address, num_peers, robust, log_degree, log_scale):
    lib.serverPhase1(GoString(server_address, len(server_address)), num_peers, robust, log_degree, 2.**log_scale)


def client_phase2(inputs, public_key, shamir_share, id, server_address, robust, log_degree, log_scale, resiliency):
    cInput = convertToGoSlice(inputs)
    lib.clientPhase2(cInput, public_key, shamir_share, id, GoString(server_address, len(server_address)), robust, log_degree, 2.**log_scale, resiliency)

def server_phase2(server_address, num_peers, robust, resiliency, log_degree, log_scale, input_length):
    res = lib.serverPhase2(GoString(server_address, len(server_address)), num_peers, robust, resiliency, log_degree, 2.**log_scale, input_length)
    res = res.decode()
    res = str.split(res, " ")[0:-1]
    res = [float(res_elem) for res_elem in res]
    return res
