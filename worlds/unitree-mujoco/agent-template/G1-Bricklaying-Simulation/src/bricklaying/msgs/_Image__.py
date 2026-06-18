from enum import auto
from typing import TYPE_CHECKING, Optional
from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

# root module import for resolving types
# import std_msgs


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Image_(idl.IdlStruct, typename="custom_msgs.msg.dds_.Image_"):
    height: types.uint32
    width: types.uint32
    data: types.sequence[types.uint8]
