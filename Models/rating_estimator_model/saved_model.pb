??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
3
Square
x"T
y"T"
Ttype:
2
	
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??

?
movie_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*+
shared_namemovie_embedding/embeddings
?
.movie_embedding/embeddings/Read/ReadVariableOpReadVariableOpmovie_embedding/embeddings*
_output_shapes
:	?*
dtype0
?
user_embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?/**
shared_nameuser_embedding/embeddings
?
-user_embedding/embeddings/Read/ReadVariableOpReadVariableOpuser_embedding/embeddings*
_output_shapes
:	?/*
dtype0
y
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	?*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
!Adam/movie_embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*2
shared_name#!Adam/movie_embedding/embeddings/m
?
5Adam/movie_embedding/embeddings/m/Read/ReadVariableOpReadVariableOp!Adam/movie_embedding/embeddings/m*
_output_shapes
:	?*
dtype0
?
 Adam/user_embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?/*1
shared_name" Adam/user_embedding/embeddings/m
?
4Adam/user_embedding/embeddings/m/Read/ReadVariableOpReadVariableOp Adam/user_embedding/embeddings/m*
_output_shapes
:	?/*
dtype0
?
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_5/kernel/m
?
)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes
:	?*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:*
dtype0
?
!Adam/movie_embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*2
shared_name#!Adam/movie_embedding/embeddings/v
?
5Adam/movie_embedding/embeddings/v/Read/ReadVariableOpReadVariableOp!Adam/movie_embedding/embeddings/v*
_output_shapes
:	?*
dtype0
?
 Adam/user_embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?/*1
shared_name" Adam/user_embedding/embeddings/v
?
4Adam/user_embedding/embeddings/v/Read/ReadVariableOpReadVariableOp Adam/user_embedding/embeddings/v*
_output_shapes
:	?/*
dtype0
?
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_5/kernel/v
?
)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes
:	?*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?=
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?=
value?=B?= B?<
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
 
b

embeddings
trainable_variables
regularization_losses
	variables
	keras_api
b

embeddings
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
R
 trainable_variables
!regularization_losses
"	variables
#	keras_api
x
$
activation

%kernel
&bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
R
+trainable_variables
,regularization_losses
-	variables
.	keras_api
x
/
activation

0kernel
1bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
R
6trainable_variables
7regularization_losses
8	variables
9	keras_api
h

:kernel
;bias
<trainable_variables
=regularization_losses
>	variables
?	keras_api
?
@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_ratem?m?%m?&m?0m?1m?:m?;m?v?v?%v?&v?0v?1v?:v?;v?
8
0
1
%2
&3
04
15
:6
;7
 
8
0
1
%2
&3
04
15
:6
;7
?
trainable_variables
Elayer_regularization_losses

Flayers
Gnon_trainable_variables
Hmetrics
regularization_losses
	variables
Ilayer_metrics
 
jh
VARIABLE_VALUEmovie_embedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
?
trainable_variables
Jlayer_regularization_losses

Klayers
Lnon_trainable_variables
Mmetrics
regularization_losses
	variables
Nlayer_metrics
ig
VARIABLE_VALUEuser_embedding/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
?
trainable_variables
Olayer_regularization_losses

Players
Qnon_trainable_variables
Rmetrics
regularization_losses
	variables
Slayer_metrics
 
 
 
?
trainable_variables
Tlayer_regularization_losses

Ulayers
Vnon_trainable_variables
Wmetrics
regularization_losses
	variables
Xlayer_metrics
 
 
 
?
 trainable_variables
Ylayer_regularization_losses

Zlayers
[non_trainable_variables
\metrics
!regularization_losses
"	variables
]layer_metrics
R
^trainable_variables
_regularization_losses
`	variables
a	keras_api
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1
 

%0
&1
?
'trainable_variables
blayer_regularization_losses

clayers
dnon_trainable_variables
emetrics
(regularization_losses
)	variables
flayer_metrics
 
 
 
?
+trainable_variables
glayer_regularization_losses

hlayers
inon_trainable_variables
jmetrics
,regularization_losses
-	variables
klayer_metrics
R
ltrainable_variables
mregularization_losses
n	variables
o	keras_api
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
?
2trainable_variables
player_regularization_losses

qlayers
rnon_trainable_variables
smetrics
3regularization_losses
4	variables
tlayer_metrics
 
 
 
?
6trainable_variables
ulayer_regularization_losses

vlayers
wnon_trainable_variables
xmetrics
7regularization_losses
8	variables
ylayer_metrics
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
 

:0
;1
?
<trainable_variables
zlayer_regularization_losses

{layers
|non_trainable_variables
}metrics
=regularization_losses
>	variables
~layer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
N
0
1
2
3
4
5
6
7
	8

9
10
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
^trainable_variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
_regularization_losses
`	variables
?layer_metrics
 

$0
 
 
 
 
 
 
 
 
 
 
 
?
ltrainable_variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
mregularization_losses
n	variables
?layer_metrics
 

/0
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
??
VARIABLE_VALUE!Adam/movie_embedding/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/user_embedding/embeddings/mVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/movie_embedding/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/user_embedding/embeddings/vVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
serving_default_movie_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
}
serving_default_user_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_movie_inputserving_default_user_inputmovie_embedding/embeddingsuser_embedding/embeddingsdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_200950
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename.movie_embedding/embeddings/Read/ReadVariableOp-user_embedding/embeddings/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp5Adam/movie_embedding/embeddings/m/Read/ReadVariableOp4Adam/user_embedding/embeddings/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp5Adam/movie_embedding/embeddings/v/Read/ReadVariableOp4Adam/user_embedding/embeddings/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_201479
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemovie_embedding/embeddingsuser_embedding/embeddingsdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount!Adam/movie_embedding/embeddings/m Adam/user_embedding/embeddings/mAdam/dense_5/kernel/mAdam/dense_5/bias/mAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/m!Adam/movie_embedding/embeddings/v Adam/user_embedding/embeddings/vAdam/dense_5/kernel/vAdam/dense_5/bias/vAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_201582??	
?

?
C__inference_dense_7_layer_call_and_return_conditional_losses_201340

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_200548

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_dropout_5_layer_call_and_return_conditional_losses_200613

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_5_layer_call_fn_201306

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2005482
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?L
?
C__inference_model_1_layer_call_and_return_conditional_losses_200570

inputs
inputs_1)
movie_embedding_200436:	?(
user_embedding_200450:	?/!
dense_5_200498:	?
dense_5_200500: 
dense_6_200530:
dense_6_200532: 
dense_7_200562:
dense_7_200564:
identity

identity_1

identity_2??dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?'movie_embedding/StatefulPartitionedCall?&user_embedding/StatefulPartitionedCall?
'movie_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsmovie_embedding_200436*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_movie_embedding_layer_call_and_return_conditional_losses_2004352)
'movie_embedding/StatefulPartitionedCall?
&user_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_1user_embedding_200450*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_user_embedding_layer_call_and_return_conditional_losses_2004492(
&user_embedding/StatefulPartitionedCall?
dot_product/PartitionedCallPartitionedCall0movie_embedding/StatefulPartitionedCall:output:0/user_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dot_product_layer_call_and_return_conditional_losses_2004762
dot_product/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall$dot_product/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2004842
flatten_1/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_5_200498dense_5_200500*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2004972!
dense_5/StatefulPartitionedCall?
+dense_5/ActivityRegularizer/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *8
f3R1
/__inference_dense_5_activity_regularizer_2004032-
+dense_5/ActivityRegularizer/PartitionedCall?
!dense_5/ActivityRegularizer/ShapeShape(dense_5/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_5/ActivityRegularizer/Shape?
/dense_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_5/ActivityRegularizer/strided_slice/stack?
1dense_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_1?
1dense_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_2?
)dense_5/ActivityRegularizer/strided_sliceStridedSlice*dense_5/ActivityRegularizer/Shape:output:08dense_5/ActivityRegularizer/strided_slice/stack:output:0:dense_5/ActivityRegularizer/strided_slice/stack_1:output:0:dense_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_5/ActivityRegularizer/strided_slice?
 dense_5/ActivityRegularizer/CastCast2dense_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_5/ActivityRegularizer/Cast?
#dense_5/ActivityRegularizer/truedivRealDiv4dense_5/ActivityRegularizer/PartitionedCall:output:0$dense_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_5/ActivityRegularizer/truediv?
dropout_4/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2005162
dropout_4/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_6_200530dense_6_200532*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2005292!
dense_6/StatefulPartitionedCall?
+dense_6/ActivityRegularizer/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *8
f3R1
/__inference_dense_6_activity_regularizer_2004162-
+dense_6/ActivityRegularizer/PartitionedCall?
!dense_6/ActivityRegularizer/ShapeShape(dense_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_6/ActivityRegularizer/Shape?
/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_6/ActivityRegularizer/strided_slice/stack?
1dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_1?
1dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_2?
)dense_6/ActivityRegularizer/strided_sliceStridedSlice*dense_6/ActivityRegularizer/Shape:output:08dense_6/ActivityRegularizer/strided_slice/stack:output:0:dense_6/ActivityRegularizer/strided_slice/stack_1:output:0:dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_6/ActivityRegularizer/strided_slice?
 dense_6/ActivityRegularizer/CastCast2dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_6/ActivityRegularizer/Cast?
#dense_6/ActivityRegularizer/truedivRealDiv4dense_6/ActivityRegularizer/PartitionedCall:output:0$dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_6/ActivityRegularizer/truediv?
dropout_5/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2005482
dropout_5/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_7_200562dense_7_200564*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_2005612!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall(^movie_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity'dense_5/ActivityRegularizer/truediv:z:0 ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall(^movie_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity'dense_6/ActivityRegularizer/truediv:z:0 ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall(^movie_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2R
'movie_embedding/StatefulPartitionedCall'movie_embedding/StatefulPartitionedCall2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_4_layer_call_fn_201267

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2005162
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_200390
movie_input

user_inputB
/model_1_movie_embedding_embedding_lookup_200311:	?A
.model_1_user_embedding_embedding_lookup_200317:	?/A
.model_1_dense_5_matmul_readvariableop_resource:	?=
/model_1_dense_5_biasadd_readvariableop_resource:@
.model_1_dense_6_matmul_readvariableop_resource:=
/model_1_dense_6_biasadd_readvariableop_resource:@
.model_1_dense_7_matmul_readvariableop_resource:=
/model_1_dense_7_biasadd_readvariableop_resource:
identity??&model_1/dense_5/BiasAdd/ReadVariableOp?%model_1/dense_5/MatMul/ReadVariableOp?&model_1/dense_6/BiasAdd/ReadVariableOp?%model_1/dense_6/MatMul/ReadVariableOp?&model_1/dense_7/BiasAdd/ReadVariableOp?%model_1/dense_7/MatMul/ReadVariableOp?(model_1/movie_embedding/embedding_lookup?'model_1/user_embedding/embedding_lookup?
model_1/movie_embedding/CastCastmovie_input*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_1/movie_embedding/Cast?
(model_1/movie_embedding/embedding_lookupResourceGather/model_1_movie_embedding_embedding_lookup_200311 model_1/movie_embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@model_1/movie_embedding/embedding_lookup/200311*+
_output_shapes
:?????????*
dtype02*
(model_1/movie_embedding/embedding_lookup?
1model_1/movie_embedding/embedding_lookup/IdentityIdentity1model_1/movie_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@model_1/movie_embedding/embedding_lookup/200311*+
_output_shapes
:?????????23
1model_1/movie_embedding/embedding_lookup/Identity?
3model_1/movie_embedding/embedding_lookup/Identity_1Identity:model_1/movie_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????25
3model_1/movie_embedding/embedding_lookup/Identity_1?
model_1/user_embedding/CastCast
user_input*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_1/user_embedding/Cast?
'model_1/user_embedding/embedding_lookupResourceGather.model_1_user_embedding_embedding_lookup_200317model_1/user_embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*A
_class7
53loc:@model_1/user_embedding/embedding_lookup/200317*+
_output_shapes
:?????????*
dtype02)
'model_1/user_embedding/embedding_lookup?
0model_1/user_embedding/embedding_lookup/IdentityIdentity0model_1/user_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@model_1/user_embedding/embedding_lookup/200317*+
_output_shapes
:?????????22
0model_1/user_embedding/embedding_lookup/Identity?
2model_1/user_embedding/embedding_lookup/Identity_1Identity9model_1/user_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????24
2model_1/user_embedding/embedding_lookup/Identity_1?
'model_1/dot_product/l2_normalize/SquareSquare<model_1/movie_embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????2)
'model_1/dot_product/l2_normalize/Square?
6model_1/dot_product/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :28
6model_1/dot_product/l2_normalize/Sum/reduction_indices?
$model_1/dot_product/l2_normalize/SumSum+model_1/dot_product/l2_normalize/Square:y:0?model_1/dot_product/l2_normalize/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2&
$model_1/dot_product/l2_normalize/Sum?
*model_1/dot_product/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+2,
*model_1/dot_product/l2_normalize/Maximum/y?
(model_1/dot_product/l2_normalize/MaximumMaximum-model_1/dot_product/l2_normalize/Sum:output:03model_1/dot_product/l2_normalize/Maximum/y:output:0*
T0*+
_output_shapes
:?????????2*
(model_1/dot_product/l2_normalize/Maximum?
&model_1/dot_product/l2_normalize/RsqrtRsqrt,model_1/dot_product/l2_normalize/Maximum:z:0*
T0*+
_output_shapes
:?????????2(
&model_1/dot_product/l2_normalize/Rsqrt?
 model_1/dot_product/l2_normalizeMul<model_1/movie_embedding/embedding_lookup/Identity_1:output:0*model_1/dot_product/l2_normalize/Rsqrt:y:0*
T0*+
_output_shapes
:?????????2"
 model_1/dot_product/l2_normalize?
)model_1/dot_product/l2_normalize_1/SquareSquare;model_1/user_embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????2+
)model_1/dot_product/l2_normalize_1/Square?
8model_1/dot_product/l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2:
8model_1/dot_product/l2_normalize_1/Sum/reduction_indices?
&model_1/dot_product/l2_normalize_1/SumSum-model_1/dot_product/l2_normalize_1/Square:y:0Amodel_1/dot_product/l2_normalize_1/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2(
&model_1/dot_product/l2_normalize_1/Sum?
,model_1/dot_product/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+2.
,model_1/dot_product/l2_normalize_1/Maximum/y?
*model_1/dot_product/l2_normalize_1/MaximumMaximum/model_1/dot_product/l2_normalize_1/Sum:output:05model_1/dot_product/l2_normalize_1/Maximum/y:output:0*
T0*+
_output_shapes
:?????????2,
*model_1/dot_product/l2_normalize_1/Maximum?
(model_1/dot_product/l2_normalize_1/RsqrtRsqrt.model_1/dot_product/l2_normalize_1/Maximum:z:0*
T0*+
_output_shapes
:?????????2*
(model_1/dot_product/l2_normalize_1/Rsqrt?
"model_1/dot_product/l2_normalize_1Mul;model_1/user_embedding/embedding_lookup/Identity_1:output:0,model_1/dot_product/l2_normalize_1/Rsqrt:y:0*
T0*+
_output_shapes
:?????????2$
"model_1/dot_product/l2_normalize_1?
"model_1/dot_product/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"model_1/dot_product/transpose/perm?
model_1/dot_product/transpose	Transpose&model_1/dot_product/l2_normalize_1:z:0+model_1/dot_product/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
model_1/dot_product/transpose?
model_1/dot_product/MatMulBatchMatMulV2$model_1/dot_product/l2_normalize:z:0!model_1/dot_product/transpose:y:0*
T0*+
_output_shapes
:?????????2
model_1/dot_product/MatMul?
model_1/dot_product/ShapeShape#model_1/dot_product/MatMul:output:0*
T0*
_output_shapes
:2
model_1/dot_product/Shape?
model_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
model_1/flatten_1/Const?
model_1/flatten_1/ReshapeReshape#model_1/dot_product/MatMul:output:0 model_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
model_1/flatten_1/Reshape?
%model_1/dense_5/MatMul/ReadVariableOpReadVariableOp.model_1_dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02'
%model_1/dense_5/MatMul/ReadVariableOp?
model_1/dense_5/MatMulMatMul"model_1/flatten_1/Reshape:output:0-model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_5/MatMul?
&model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_1/dense_5/BiasAdd/ReadVariableOp?
model_1/dense_5/BiasAddBiasAdd model_1/dense_5/MatMul:product:0.model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_5/BiasAdd?
model_1/dense_5/re_lu_4/ReluRelu model_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_5/re_lu_4/Relu?
*model_1/dense_5/ActivityRegularizer/SquareSquare*model_1/dense_5/re_lu_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2,
*model_1/dense_5/ActivityRegularizer/Square?
)model_1/dense_5/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)model_1/dense_5/ActivityRegularizer/Const?
'model_1/dense_5/ActivityRegularizer/SumSum.model_1/dense_5/ActivityRegularizer/Square:y:02model_1/dense_5/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2)
'model_1/dense_5/ActivityRegularizer/Sum?
)model_1/dense_5/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)model_1/dense_5/ActivityRegularizer/mul/x?
'model_1/dense_5/ActivityRegularizer/mulMul2model_1/dense_5/ActivityRegularizer/mul/x:output:00model_1/dense_5/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'model_1/dense_5/ActivityRegularizer/mul?
)model_1/dense_5/ActivityRegularizer/ShapeShape*model_1/dense_5/re_lu_4/Relu:activations:0*
T0*
_output_shapes
:2+
)model_1/dense_5/ActivityRegularizer/Shape?
7model_1/dense_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7model_1/dense_5/ActivityRegularizer/strided_slice/stack?
9model_1/dense_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_1/dense_5/ActivityRegularizer/strided_slice/stack_1?
9model_1/dense_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_1/dense_5/ActivityRegularizer/strided_slice/stack_2?
1model_1/dense_5/ActivityRegularizer/strided_sliceStridedSlice2model_1/dense_5/ActivityRegularizer/Shape:output:0@model_1/dense_5/ActivityRegularizer/strided_slice/stack:output:0Bmodel_1/dense_5/ActivityRegularizer/strided_slice/stack_1:output:0Bmodel_1/dense_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1model_1/dense_5/ActivityRegularizer/strided_slice?
(model_1/dense_5/ActivityRegularizer/CastCast:model_1/dense_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(model_1/dense_5/ActivityRegularizer/Cast?
+model_1/dense_5/ActivityRegularizer/truedivRealDiv+model_1/dense_5/ActivityRegularizer/mul:z:0,model_1/dense_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2-
+model_1/dense_5/ActivityRegularizer/truediv?
model_1/dropout_4/IdentityIdentity*model_1/dense_5/re_lu_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2
model_1/dropout_4/Identity?
%model_1/dense_6/MatMul/ReadVariableOpReadVariableOp.model_1_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%model_1/dense_6/MatMul/ReadVariableOp?
model_1/dense_6/MatMulMatMul#model_1/dropout_4/Identity:output:0-model_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_6/MatMul?
&model_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_1/dense_6/BiasAdd/ReadVariableOp?
model_1/dense_6/BiasAddBiasAdd model_1/dense_6/MatMul:product:0.model_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_6/BiasAdd?
model_1/dense_6/re_lu_5/ReluRelu model_1/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_6/re_lu_5/Relu?
*model_1/dense_6/ActivityRegularizer/SquareSquare*model_1/dense_6/re_lu_5/Relu:activations:0*
T0*'
_output_shapes
:?????????2,
*model_1/dense_6/ActivityRegularizer/Square?
)model_1/dense_6/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2+
)model_1/dense_6/ActivityRegularizer/Const?
'model_1/dense_6/ActivityRegularizer/SumSum.model_1/dense_6/ActivityRegularizer/Square:y:02model_1/dense_6/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2)
'model_1/dense_6/ActivityRegularizer/Sum?
)model_1/dense_6/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)model_1/dense_6/ActivityRegularizer/mul/x?
'model_1/dense_6/ActivityRegularizer/mulMul2model_1/dense_6/ActivityRegularizer/mul/x:output:00model_1/dense_6/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'model_1/dense_6/ActivityRegularizer/mul?
)model_1/dense_6/ActivityRegularizer/ShapeShape*model_1/dense_6/re_lu_5/Relu:activations:0*
T0*
_output_shapes
:2+
)model_1/dense_6/ActivityRegularizer/Shape?
7model_1/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7model_1/dense_6/ActivityRegularizer/strided_slice/stack?
9model_1/dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_1/dense_6/ActivityRegularizer/strided_slice/stack_1?
9model_1/dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9model_1/dense_6/ActivityRegularizer/strided_slice/stack_2?
1model_1/dense_6/ActivityRegularizer/strided_sliceStridedSlice2model_1/dense_6/ActivityRegularizer/Shape:output:0@model_1/dense_6/ActivityRegularizer/strided_slice/stack:output:0Bmodel_1/dense_6/ActivityRegularizer/strided_slice/stack_1:output:0Bmodel_1/dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1model_1/dense_6/ActivityRegularizer/strided_slice?
(model_1/dense_6/ActivityRegularizer/CastCast:model_1/dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2*
(model_1/dense_6/ActivityRegularizer/Cast?
+model_1/dense_6/ActivityRegularizer/truedivRealDiv+model_1/dense_6/ActivityRegularizer/mul:z:0,model_1/dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2-
+model_1/dense_6/ActivityRegularizer/truediv?
model_1/dropout_5/IdentityIdentity*model_1/dense_6/re_lu_5/Relu:activations:0*
T0*'
_output_shapes
:?????????2
model_1/dropout_5/Identity?
%model_1/dense_7/MatMul/ReadVariableOpReadVariableOp.model_1_dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%model_1/dense_7/MatMul/ReadVariableOp?
model_1/dense_7/MatMulMatMul#model_1/dropout_5/Identity:output:0-model_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_7/MatMul?
&model_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_1/dense_7/BiasAdd/ReadVariableOp?
model_1/dense_7/BiasAddBiasAdd model_1/dense_7/MatMul:product:0.model_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_7/BiasAdd?
model_1/dense_7/SigmoidSigmoid model_1/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_7/Sigmoid?
IdentityIdentitymodel_1/dense_7/Sigmoid:y:0'^model_1/dense_5/BiasAdd/ReadVariableOp&^model_1/dense_5/MatMul/ReadVariableOp'^model_1/dense_6/BiasAdd/ReadVariableOp&^model_1/dense_6/MatMul/ReadVariableOp'^model_1/dense_7/BiasAdd/ReadVariableOp&^model_1/dense_7/MatMul/ReadVariableOp)^model_1/movie_embedding/embedding_lookup(^model_1/user_embedding/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 2P
&model_1/dense_5/BiasAdd/ReadVariableOp&model_1/dense_5/BiasAdd/ReadVariableOp2N
%model_1/dense_5/MatMul/ReadVariableOp%model_1/dense_5/MatMul/ReadVariableOp2P
&model_1/dense_6/BiasAdd/ReadVariableOp&model_1/dense_6/BiasAdd/ReadVariableOp2N
%model_1/dense_6/MatMul/ReadVariableOp%model_1/dense_6/MatMul/ReadVariableOp2P
&model_1/dense_7/BiasAdd/ReadVariableOp&model_1/dense_7/BiasAdd/ReadVariableOp2N
%model_1/dense_7/MatMul/ReadVariableOp%model_1/dense_7/MatMul/ReadVariableOp2T
(model_1/movie_embedding/embedding_lookup(model_1/movie_embedding/embedding_lookup2R
'model_1/user_embedding/embedding_lookup'model_1/user_embedding/embedding_lookup:T P
'
_output_shapes
:?????????
%
_user_specified_namemovie_input:SO
'
_output_shapes
:?????????
$
_user_specified_name
user_input
?F
?
__inference__traced_save_201479
file_prefix9
5savev2_movie_embedding_embeddings_read_readvariableop8
4savev2_user_embedding_embeddings_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop@
<savev2_adam_movie_embedding_embeddings_m_read_readvariableop?
;savev2_adam_user_embedding_embeddings_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop@
<savev2_adam_movie_embedding_embeddings_v_read_readvariableop?
;savev2_adam_user_embedding_embeddings_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_movie_embedding_embeddings_read_readvariableop4savev2_user_embedding_embeddings_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop<savev2_adam_movie_embedding_embeddings_m_read_readvariableop;savev2_adam_user_embedding_embeddings_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop<savev2_adam_movie_embedding_embeddings_v_read_readvariableop;savev2_adam_user_embedding_embeddings_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?:	?/:	?:::::: : : : : : : :	?:	?/:	?::::::	?:	?/:	?:::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:%!

_output_shapes
:	?/:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:%!

_output_shapes
:	?/:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	?:%!

_output_shapes
:	?/:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
: 
?

?
K__inference_movie_embedding_layer_call_and_return_conditional_losses_201185

inputs*
embedding_lookup_201179:	?
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_201179Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0**
_class 
loc:@embedding_lookup/201179*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@embedding_lookup/201179*+
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_dense_7_layer_call_and_return_conditional_losses_200561

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
X
,__inference_dot_product_layer_call_fn_201208
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dot_product_layer_call_and_return_conditional_losses_2004762
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????:?????????:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

?
$__inference_signature_wrapper_200950
movie_input

user_input
unknown:	?
	unknown_0:	?/
	unknown_1:	?
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmovie_input
user_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_2003902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_namemovie_input:SO
'
_output_shapes
:?????????
$
_user_specified_name
user_input
?

?
C__inference_dense_6_layer_call_and_return_conditional_losses_200529

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddh
re_lu_5/ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
re_lu_5/Relu?
IdentityIdentityre_lu_5/Relu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_201316

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
(__inference_model_1_layer_call_fn_200824
movie_input

user_input
unknown:	?
	unknown_0:	?/
	unknown_1:	?
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmovie_input
user_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????: : **
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2007792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_namemovie_input:SO
'
_output_shapes
:?????????
$
_user_specified_name
user_input
?
?
(__inference_dense_5_layer_call_fn_201251

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2004972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_7_layer_call_fn_201329

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_2005612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_dense_6_layer_call_and_return_all_conditional_losses_201301

inputs
unknown:
	unknown_0:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2005292
StatefulPartitionedCall?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *8
f3R1
/__inference_dense_6_activity_regularizer_2004162
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?L
?
C__inference_model_1_layer_call_and_return_conditional_losses_200779

inputs
inputs_1)
movie_embedding_200735:	?(
user_embedding_200738:	?/!
dense_5_200743:	?
dense_5_200745: 
dense_6_200757:
dense_6_200759: 
dense_7_200771:
dense_7_200773:
identity

identity_1

identity_2??dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?'movie_embedding/StatefulPartitionedCall?&user_embedding/StatefulPartitionedCall?
'movie_embedding/StatefulPartitionedCallStatefulPartitionedCallinputsmovie_embedding_200735*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_movie_embedding_layer_call_and_return_conditional_losses_2004352)
'movie_embedding/StatefulPartitionedCall?
&user_embedding/StatefulPartitionedCallStatefulPartitionedCallinputs_1user_embedding_200738*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_user_embedding_layer_call_and_return_conditional_losses_2004492(
&user_embedding/StatefulPartitionedCall?
dot_product/PartitionedCallPartitionedCall0movie_embedding/StatefulPartitionedCall:output:0/user_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dot_product_layer_call_and_return_conditional_losses_2004762
dot_product/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall$dot_product/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2004842
flatten_1/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_5_200743dense_5_200745*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2004972!
dense_5/StatefulPartitionedCall?
+dense_5/ActivityRegularizer/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *8
f3R1
/__inference_dense_5_activity_regularizer_2004032-
+dense_5/ActivityRegularizer/PartitionedCall?
!dense_5/ActivityRegularizer/ShapeShape(dense_5/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_5/ActivityRegularizer/Shape?
/dense_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_5/ActivityRegularizer/strided_slice/stack?
1dense_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_1?
1dense_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_2?
)dense_5/ActivityRegularizer/strided_sliceStridedSlice*dense_5/ActivityRegularizer/Shape:output:08dense_5/ActivityRegularizer/strided_slice/stack:output:0:dense_5/ActivityRegularizer/strided_slice/stack_1:output:0:dense_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_5/ActivityRegularizer/strided_slice?
 dense_5/ActivityRegularizer/CastCast2dense_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_5/ActivityRegularizer/Cast?
#dense_5/ActivityRegularizer/truedivRealDiv4dense_5/ActivityRegularizer/PartitionedCall:output:0$dense_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_5/ActivityRegularizer/truediv?
dropout_4/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2006502
dropout_4/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_6_200757dense_6_200759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2005292!
dense_6/StatefulPartitionedCall?
+dense_6/ActivityRegularizer/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *8
f3R1
/__inference_dense_6_activity_regularizer_2004162-
+dense_6/ActivityRegularizer/PartitionedCall?
!dense_6/ActivityRegularizer/ShapeShape(dense_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_6/ActivityRegularizer/Shape?
/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_6/ActivityRegularizer/strided_slice/stack?
1dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_1?
1dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_2?
)dense_6/ActivityRegularizer/strided_sliceStridedSlice*dense_6/ActivityRegularizer/Shape:output:08dense_6/ActivityRegularizer/strided_slice/stack:output:0:dense_6/ActivityRegularizer/strided_slice/stack_1:output:0:dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_6/ActivityRegularizer/strided_slice?
 dense_6/ActivityRegularizer/CastCast2dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_6/ActivityRegularizer/Cast?
#dense_6/ActivityRegularizer/truedivRealDiv4dense_6/ActivityRegularizer/PartitionedCall:output:0$dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_6/ActivityRegularizer/truediv?
dropout_5/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2006132
dropout_5/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_7_200771dense_7_200773*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_2005612!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall(^movie_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity'dense_5/ActivityRegularizer/truediv:z:0 ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall(^movie_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity'dense_6/ActivityRegularizer/truediv:z:0 ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall(^movie_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2R
'movie_embedding/StatefulPartitionedCall'movie_embedding/StatefulPartitionedCall2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_dropout_4_layer_call_and_return_conditional_losses_200650

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_dense_5_layer_call_and_return_all_conditional_losses_201262

inputs
unknown:	?
	unknown_0:
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2004972
StatefulPartitionedCall?
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *8
f3R1
/__inference_dense_5_activity_regularizer_2004032
PartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_200484

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
J__inference_user_embedding_layer_call_and_return_conditional_losses_200449

inputs*
embedding_lookup_200443:	?/
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_200443Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0**
_class 
loc:@embedding_lookup/200443*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@embedding_lookup/200443*+
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_flatten_1_layer_call_fn_201236

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2004842
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_dense_6_layer_call_and_return_conditional_losses_201362

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddh
re_lu_5/ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
re_lu_5/Relu?
IdentityIdentityre_lu_5/Relu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?L
?
C__inference_model_1_layer_call_and_return_conditional_losses_200920
movie_input

user_input)
movie_embedding_200876:	?(
user_embedding_200879:	?/!
dense_5_200884:	?
dense_5_200886: 
dense_6_200898:
dense_6_200900: 
dense_7_200912:
dense_7_200914:
identity

identity_1

identity_2??dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?'movie_embedding/StatefulPartitionedCall?&user_embedding/StatefulPartitionedCall?
'movie_embedding/StatefulPartitionedCallStatefulPartitionedCallmovie_inputmovie_embedding_200876*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_movie_embedding_layer_call_and_return_conditional_losses_2004352)
'movie_embedding/StatefulPartitionedCall?
&user_embedding/StatefulPartitionedCallStatefulPartitionedCall
user_inputuser_embedding_200879*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_user_embedding_layer_call_and_return_conditional_losses_2004492(
&user_embedding/StatefulPartitionedCall?
dot_product/PartitionedCallPartitionedCall0movie_embedding/StatefulPartitionedCall:output:0/user_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dot_product_layer_call_and_return_conditional_losses_2004762
dot_product/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall$dot_product/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2004842
flatten_1/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_5_200884dense_5_200886*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2004972!
dense_5/StatefulPartitionedCall?
+dense_5/ActivityRegularizer/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *8
f3R1
/__inference_dense_5_activity_regularizer_2004032-
+dense_5/ActivityRegularizer/PartitionedCall?
!dense_5/ActivityRegularizer/ShapeShape(dense_5/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_5/ActivityRegularizer/Shape?
/dense_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_5/ActivityRegularizer/strided_slice/stack?
1dense_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_1?
1dense_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_2?
)dense_5/ActivityRegularizer/strided_sliceStridedSlice*dense_5/ActivityRegularizer/Shape:output:08dense_5/ActivityRegularizer/strided_slice/stack:output:0:dense_5/ActivityRegularizer/strided_slice/stack_1:output:0:dense_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_5/ActivityRegularizer/strided_slice?
 dense_5/ActivityRegularizer/CastCast2dense_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_5/ActivityRegularizer/Cast?
#dense_5/ActivityRegularizer/truedivRealDiv4dense_5/ActivityRegularizer/PartitionedCall:output:0$dense_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_5/ActivityRegularizer/truediv?
dropout_4/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2006502
dropout_4/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_6_200898dense_6_200900*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2005292!
dense_6/StatefulPartitionedCall?
+dense_6/ActivityRegularizer/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *8
f3R1
/__inference_dense_6_activity_regularizer_2004162-
+dense_6/ActivityRegularizer/PartitionedCall?
!dense_6/ActivityRegularizer/ShapeShape(dense_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_6/ActivityRegularizer/Shape?
/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_6/ActivityRegularizer/strided_slice/stack?
1dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_1?
1dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_2?
)dense_6/ActivityRegularizer/strided_sliceStridedSlice*dense_6/ActivityRegularizer/Shape:output:08dense_6/ActivityRegularizer/strided_slice/stack:output:0:dense_6/ActivityRegularizer/strided_slice/stack_1:output:0:dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_6/ActivityRegularizer/strided_slice?
 dense_6/ActivityRegularizer/CastCast2dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_6/ActivityRegularizer/Cast?
#dense_6/ActivityRegularizer/truedivRealDiv4dense_6/ActivityRegularizer/PartitionedCall:output:0$dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_6/ActivityRegularizer/truediv?
dropout_5/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2006132
dropout_5/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_7_200912dense_7_200914*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_2005612!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall(^movie_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity'dense_5/ActivityRegularizer/truediv:z:0 ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall(^movie_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity'dense_6/ActivityRegularizer/truediv:z:0 ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall(^movie_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2R
'movie_embedding/StatefulPartitionedCall'movie_embedding/StatefulPartitionedCall2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_namemovie_input:SO
'
_output_shapes
:?????????
$
_user_specified_name
user_input
?

?
C__inference_dense_5_layer_call_and_return_conditional_losses_200497

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddh
re_lu_4/ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
re_lu_4/Relu?
IdentityIdentityre_lu_4/Relu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
(__inference_model_1_layer_call_fn_200998
inputs_0
inputs_1
unknown:	?
	unknown_0:	?/
	unknown_1:	?
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????: : **
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2007792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
q
G__inference_dot_product_layer_call_and_return_conditional_losses_200476

inputs
inputs_1
identityr
l2_normalize/SquareSquareinputs*
T0*+
_output_shapes
:?????????2
l2_normalize/Square?
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2$
"l2_normalize/Sum/reduction_indices?
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2
l2_normalize/Sumu
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+2
l2_normalize/Maximum/y?
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*+
_output_shapes
:?????????2
l2_normalize/Maximum?
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*+
_output_shapes
:?????????2
l2_normalize/Rsqrty
l2_normalizeMulinputsl2_normalize/Rsqrt:y:0*
T0*+
_output_shapes
:?????????2
l2_normalizex
l2_normalize_1/SquareSquareinputs_1*
T0*+
_output_shapes
:?????????2
l2_normalize_1/Square?
$l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2&
$l2_normalize_1/Sum/reduction_indices?
l2_normalize_1/SumSuml2_normalize_1/Square:y:0-l2_normalize_1/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2
l2_normalize_1/Sumy
l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+2
l2_normalize_1/Maximum/y?
l2_normalize_1/MaximumMaximuml2_normalize_1/Sum:output:0!l2_normalize_1/Maximum/y:output:0*
T0*+
_output_shapes
:?????????2
l2_normalize_1/Maximum?
l2_normalize_1/RsqrtRsqrtl2_normalize_1/Maximum:z:0*
T0*+
_output_shapes
:?????????2
l2_normalize_1/Rsqrt?
l2_normalize_1Mulinputs_1l2_normalize_1/Rsqrt:y:0*
T0*+
_output_shapes
:?????????2
l2_normalize_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposel2_normalize_1:z:0transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposex
MatMulBatchMatMulV2l2_normalize:z:0transpose:y:0*
T0*+
_output_shapes
:?????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shapeg
IdentityIdentityMatMul:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?L
?
C__inference_model_1_layer_call_and_return_conditional_losses_200872
movie_input

user_input)
movie_embedding_200828:	?(
user_embedding_200831:	?/!
dense_5_200836:	?
dense_5_200838: 
dense_6_200850:
dense_6_200852: 
dense_7_200864:
dense_7_200866:
identity

identity_1

identity_2??dense_5/StatefulPartitionedCall?dense_6/StatefulPartitionedCall?dense_7/StatefulPartitionedCall?'movie_embedding/StatefulPartitionedCall?&user_embedding/StatefulPartitionedCall?
'movie_embedding/StatefulPartitionedCallStatefulPartitionedCallmovie_inputmovie_embedding_200828*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_movie_embedding_layer_call_and_return_conditional_losses_2004352)
'movie_embedding/StatefulPartitionedCall?
&user_embedding/StatefulPartitionedCallStatefulPartitionedCall
user_inputuser_embedding_200831*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_user_embedding_layer_call_and_return_conditional_losses_2004492(
&user_embedding/StatefulPartitionedCall?
dot_product/PartitionedCallPartitionedCall0movie_embedding/StatefulPartitionedCall:output:0/user_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dot_product_layer_call_and_return_conditional_losses_2004762
dot_product/PartitionedCall?
flatten_1/PartitionedCallPartitionedCall$dot_product/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_2004842
flatten_1/PartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_5_200836dense_5_200838*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2004972!
dense_5/StatefulPartitionedCall?
+dense_5/ActivityRegularizer/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *8
f3R1
/__inference_dense_5_activity_regularizer_2004032-
+dense_5/ActivityRegularizer/PartitionedCall?
!dense_5/ActivityRegularizer/ShapeShape(dense_5/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_5/ActivityRegularizer/Shape?
/dense_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_5/ActivityRegularizer/strided_slice/stack?
1dense_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_1?
1dense_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_2?
)dense_5/ActivityRegularizer/strided_sliceStridedSlice*dense_5/ActivityRegularizer/Shape:output:08dense_5/ActivityRegularizer/strided_slice/stack:output:0:dense_5/ActivityRegularizer/strided_slice/stack_1:output:0:dense_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_5/ActivityRegularizer/strided_slice?
 dense_5/ActivityRegularizer/CastCast2dense_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_5/ActivityRegularizer/Cast?
#dense_5/ActivityRegularizer/truedivRealDiv4dense_5/ActivityRegularizer/PartitionedCall:output:0$dense_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_5/ActivityRegularizer/truediv?
dropout_4/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2005162
dropout_4/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_6_200850dense_6_200852*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2005292!
dense_6/StatefulPartitionedCall?
+dense_6/ActivityRegularizer/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *8
f3R1
/__inference_dense_6_activity_regularizer_2004162-
+dense_6/ActivityRegularizer/PartitionedCall?
!dense_6/ActivityRegularizer/ShapeShape(dense_6/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_6/ActivityRegularizer/Shape?
/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_6/ActivityRegularizer/strided_slice/stack?
1dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_1?
1dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_2?
)dense_6/ActivityRegularizer/strided_sliceStridedSlice*dense_6/ActivityRegularizer/Shape:output:08dense_6/ActivityRegularizer/strided_slice/stack:output:0:dense_6/ActivityRegularizer/strided_slice/stack_1:output:0:dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_6/ActivityRegularizer/strided_slice?
 dense_6/ActivityRegularizer/CastCast2dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_6/ActivityRegularizer/Cast?
#dense_6/ActivityRegularizer/truedivRealDiv4dense_6/ActivityRegularizer/PartitionedCall:output:0$dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_6/ActivityRegularizer/truediv?
dropout_5/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2005482
dropout_5/PartitionedCall?
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_7_200864dense_7_200866*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_2005612!
dense_7/StatefulPartitionedCall?
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall(^movie_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity'dense_5/ActivityRegularizer/truediv:z:0 ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall(^movie_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity'dense_6/ActivityRegularizer/truediv:z:0 ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall(^movie_embedding/StatefulPartitionedCall'^user_embedding/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2R
'movie_embedding/StatefulPartitionedCall'movie_embedding/StatefulPartitionedCall2P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_namemovie_input:SO
'
_output_shapes
:?????????
$
_user_specified_name
user_input
?z
?
C__inference_model_1_layer_call_and_return_conditional_losses_201084
inputs_0
inputs_1:
'movie_embedding_embedding_lookup_201003:	?9
&user_embedding_embedding_lookup_201009:	?/9
&dense_5_matmul_readvariableop_resource:	?5
'dense_5_biasadd_readvariableop_resource:8
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:
identity

identity_1

identity_2??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp? movie_embedding/embedding_lookup?user_embedding/embedding_lookup
movie_embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
movie_embedding/Cast?
 movie_embedding/embedding_lookupResourceGather'movie_embedding_embedding_lookup_201003movie_embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@movie_embedding/embedding_lookup/201003*+
_output_shapes
:?????????*
dtype02"
 movie_embedding/embedding_lookup?
)movie_embedding/embedding_lookup/IdentityIdentity)movie_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@movie_embedding/embedding_lookup/201003*+
_output_shapes
:?????????2+
)movie_embedding/embedding_lookup/Identity?
+movie_embedding/embedding_lookup/Identity_1Identity2movie_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2-
+movie_embedding/embedding_lookup/Identity_1}
user_embedding/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:?????????2
user_embedding/Cast?
user_embedding/embedding_lookupResourceGather&user_embedding_embedding_lookup_201009user_embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@user_embedding/embedding_lookup/201009*+
_output_shapes
:?????????*
dtype02!
user_embedding/embedding_lookup?
(user_embedding/embedding_lookup/IdentityIdentity(user_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@user_embedding/embedding_lookup/201009*+
_output_shapes
:?????????2*
(user_embedding/embedding_lookup/Identity?
*user_embedding/embedding_lookup/Identity_1Identity1user_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2,
*user_embedding/embedding_lookup/Identity_1?
dot_product/l2_normalize/SquareSquare4movie_embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????2!
dot_product/l2_normalize/Square?
.dot_product/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.dot_product/l2_normalize/Sum/reduction_indices?
dot_product/l2_normalize/SumSum#dot_product/l2_normalize/Square:y:07dot_product/l2_normalize/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2
dot_product/l2_normalize/Sum?
"dot_product/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+2$
"dot_product/l2_normalize/Maximum/y?
 dot_product/l2_normalize/MaximumMaximum%dot_product/l2_normalize/Sum:output:0+dot_product/l2_normalize/Maximum/y:output:0*
T0*+
_output_shapes
:?????????2"
 dot_product/l2_normalize/Maximum?
dot_product/l2_normalize/RsqrtRsqrt$dot_product/l2_normalize/Maximum:z:0*
T0*+
_output_shapes
:?????????2 
dot_product/l2_normalize/Rsqrt?
dot_product/l2_normalizeMul4movie_embedding/embedding_lookup/Identity_1:output:0"dot_product/l2_normalize/Rsqrt:y:0*
T0*+
_output_shapes
:?????????2
dot_product/l2_normalize?
!dot_product/l2_normalize_1/SquareSquare3user_embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????2#
!dot_product/l2_normalize_1/Square?
0dot_product/l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :22
0dot_product/l2_normalize_1/Sum/reduction_indices?
dot_product/l2_normalize_1/SumSum%dot_product/l2_normalize_1/Square:y:09dot_product/l2_normalize_1/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2 
dot_product/l2_normalize_1/Sum?
$dot_product/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+2&
$dot_product/l2_normalize_1/Maximum/y?
"dot_product/l2_normalize_1/MaximumMaximum'dot_product/l2_normalize_1/Sum:output:0-dot_product/l2_normalize_1/Maximum/y:output:0*
T0*+
_output_shapes
:?????????2$
"dot_product/l2_normalize_1/Maximum?
 dot_product/l2_normalize_1/RsqrtRsqrt&dot_product/l2_normalize_1/Maximum:z:0*
T0*+
_output_shapes
:?????????2"
 dot_product/l2_normalize_1/Rsqrt?
dot_product/l2_normalize_1Mul3user_embedding/embedding_lookup/Identity_1:output:0$dot_product/l2_normalize_1/Rsqrt:y:0*
T0*+
_output_shapes
:?????????2
dot_product/l2_normalize_1?
dot_product/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
dot_product/transpose/perm?
dot_product/transpose	Transposedot_product/l2_normalize_1:z:0#dot_product/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
dot_product/transpose?
dot_product/MatMulBatchMatMulV2dot_product/l2_normalize:z:0dot_product/transpose:y:0*
T0*+
_output_shapes
:?????????2
dot_product/MatMulq
dot_product/ShapeShapedot_product/MatMul:output:0*
T0*
_output_shapes
:2
dot_product/Shapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_1/Const?
flatten_1/ReshapeReshapedot_product/MatMul:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshape?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulflatten_1/Reshape:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAdd?
dense_5/re_lu_4/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_5/re_lu_4/Relu?
"dense_5/ActivityRegularizer/SquareSquare"dense_5/re_lu_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2$
"dense_5/ActivityRegularizer/Square?
!dense_5/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_5/ActivityRegularizer/Const?
dense_5/ActivityRegularizer/SumSum&dense_5/ActivityRegularizer/Square:y:0*dense_5/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_5/ActivityRegularizer/Sum?
!dense_5/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_5/ActivityRegularizer/mul/x?
dense_5/ActivityRegularizer/mulMul*dense_5/ActivityRegularizer/mul/x:output:0(dense_5/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_5/ActivityRegularizer/mul?
!dense_5/ActivityRegularizer/ShapeShape"dense_5/re_lu_4/Relu:activations:0*
T0*
_output_shapes
:2#
!dense_5/ActivityRegularizer/Shape?
/dense_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_5/ActivityRegularizer/strided_slice/stack?
1dense_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_1?
1dense_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_2?
)dense_5/ActivityRegularizer/strided_sliceStridedSlice*dense_5/ActivityRegularizer/Shape:output:08dense_5/ActivityRegularizer/strided_slice/stack:output:0:dense_5/ActivityRegularizer/strided_slice/stack_1:output:0:dense_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_5/ActivityRegularizer/strided_slice?
 dense_5/ActivityRegularizer/CastCast2dense_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_5/ActivityRegularizer/Cast?
#dense_5/ActivityRegularizer/truedivRealDiv#dense_5/ActivityRegularizer/mul:z:0$dense_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_5/ActivityRegularizer/truediv?
dropout_4/IdentityIdentity"dense_5/re_lu_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2
dropout_4/Identity?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMuldropout_4/Identity:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAdd?
dense_6/re_lu_5/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_6/re_lu_5/Relu?
"dense_6/ActivityRegularizer/SquareSquare"dense_6/re_lu_5/Relu:activations:0*
T0*'
_output_shapes
:?????????2$
"dense_6/ActivityRegularizer/Square?
!dense_6/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_6/ActivityRegularizer/Const?
dense_6/ActivityRegularizer/SumSum&dense_6/ActivityRegularizer/Square:y:0*dense_6/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_6/ActivityRegularizer/Sum?
!dense_6/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_6/ActivityRegularizer/mul/x?
dense_6/ActivityRegularizer/mulMul*dense_6/ActivityRegularizer/mul/x:output:0(dense_6/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_6/ActivityRegularizer/mul?
!dense_6/ActivityRegularizer/ShapeShape"dense_6/re_lu_5/Relu:activations:0*
T0*
_output_shapes
:2#
!dense_6/ActivityRegularizer/Shape?
/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_6/ActivityRegularizer/strided_slice/stack?
1dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_1?
1dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_2?
)dense_6/ActivityRegularizer/strided_sliceStridedSlice*dense_6/ActivityRegularizer/Shape:output:08dense_6/ActivityRegularizer/strided_slice/stack:output:0:dense_6/ActivityRegularizer/strided_slice/stack_1:output:0:dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_6/ActivityRegularizer/strided_slice?
 dense_6/ActivityRegularizer/CastCast2dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_6/ActivityRegularizer/Cast?
#dense_6/ActivityRegularizer/truedivRealDiv#dense_6/ActivityRegularizer/mul:z:0$dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_6/ActivityRegularizer/truediv?
dropout_5/IdentityIdentity"dense_6/re_lu_5/Relu:activations:0*
T0*'
_output_shapes
:?????????2
dropout_5/Identity?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMuldropout_5/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/BiasAddy
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_7/Sigmoid?
IdentityIdentitydense_7/Sigmoid:y:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp!^movie_embedding/embedding_lookup ^user_embedding/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity'dense_5/ActivityRegularizer/truediv:z:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp!^movie_embedding/embedding_lookup ^user_embedding/embedding_lookup*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity'dense_6/ActivityRegularizer/truediv:z:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp!^movie_embedding/embedding_lookup ^user_embedding/embedding_lookup*
T0*
_output_shapes
: 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2D
 movie_embedding/embedding_lookup movie_embedding/embedding_lookup2B
user_embedding/embedding_lookupuser_embedding/embedding_lookup:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
0__inference_movie_embedding_layer_call_fn_201175

inputs
unknown:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_movie_embedding_layer_call_and_return_conditional_losses_2004352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
s
G__inference_dot_product_layer_call_and_return_conditional_losses_201231
inputs_0
inputs_1
identityt
l2_normalize/SquareSquareinputs_0*
T0*+
_output_shapes
:?????????2
l2_normalize/Square?
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2$
"l2_normalize/Sum/reduction_indices?
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2
l2_normalize/Sumu
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+2
l2_normalize/Maximum/y?
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*+
_output_shapes
:?????????2
l2_normalize/Maximum?
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*+
_output_shapes
:?????????2
l2_normalize/Rsqrt{
l2_normalizeMulinputs_0l2_normalize/Rsqrt:y:0*
T0*+
_output_shapes
:?????????2
l2_normalizex
l2_normalize_1/SquareSquareinputs_1*
T0*+
_output_shapes
:?????????2
l2_normalize_1/Square?
$l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2&
$l2_normalize_1/Sum/reduction_indices?
l2_normalize_1/SumSuml2_normalize_1/Square:y:0-l2_normalize_1/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2
l2_normalize_1/Sumy
l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+2
l2_normalize_1/Maximum/y?
l2_normalize_1/MaximumMaximuml2_normalize_1/Sum:output:0!l2_normalize_1/Maximum/y:output:0*
T0*+
_output_shapes
:?????????2
l2_normalize_1/Maximum?
l2_normalize_1/RsqrtRsqrtl2_normalize_1/Maximum:z:0*
T0*+
_output_shapes
:?????????2
l2_normalize_1/Rsqrt?
l2_normalize_1Mulinputs_1l2_normalize_1/Rsqrt:y:0*
T0*+
_output_shapes
:?????????2
l2_normalize_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposel2_normalize_1:z:0transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposex
MatMulBatchMatMulV2l2_normalize:z:0transpose:y:0*
T0*+
_output_shapes
:?????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shapeg
IdentityIdentityMatMul:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:?????????:?????????:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
(__inference_dense_6_layer_call_fn_201290

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_2005292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_201242

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_dropout_4_layer_call_and_return_conditional_losses_201281

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
(__inference_model_1_layer_call_fn_200591
movie_input

user_input
unknown:	?
	unknown_0:	?/
	unknown_1:	?
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmovie_input
user_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????: : **
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2005702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_namemovie_input:SO
'
_output_shapes
:?????????
$
_user_specified_name
user_input
?
F
*__inference_dropout_5_layer_call_fn_201311

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2006132
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
(__inference_model_1_layer_call_fn_200974
inputs_0
inputs_1
unknown:	?
	unknown_0:	?/
	unknown_1:	?
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????: : **
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2005702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
a
E__inference_dropout_5_layer_call_and_return_conditional_losses_201320

inputs
identityZ
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?x
?
C__inference_model_1_layer_call_and_return_conditional_losses_201168
inputs_0
inputs_1:
'movie_embedding_embedding_lookup_201089:	?9
&user_embedding_embedding_lookup_201095:	?/9
&dense_5_matmul_readvariableop_resource:	?5
'dense_5_biasadd_readvariableop_resource:8
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:5
'dense_7_biasadd_readvariableop_resource:
identity

identity_1

identity_2??dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?dense_6/BiasAdd/ReadVariableOp?dense_6/MatMul/ReadVariableOp?dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp? movie_embedding/embedding_lookup?user_embedding/embedding_lookup
movie_embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
movie_embedding/Cast?
 movie_embedding/embedding_lookupResourceGather'movie_embedding_embedding_lookup_201089movie_embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@movie_embedding/embedding_lookup/201089*+
_output_shapes
:?????????*
dtype02"
 movie_embedding/embedding_lookup?
)movie_embedding/embedding_lookup/IdentityIdentity)movie_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@movie_embedding/embedding_lookup/201089*+
_output_shapes
:?????????2+
)movie_embedding/embedding_lookup/Identity?
+movie_embedding/embedding_lookup/Identity_1Identity2movie_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2-
+movie_embedding/embedding_lookup/Identity_1}
user_embedding/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:?????????2
user_embedding/Cast?
user_embedding/embedding_lookupResourceGather&user_embedding_embedding_lookup_201095user_embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*9
_class/
-+loc:@user_embedding/embedding_lookup/201095*+
_output_shapes
:?????????*
dtype02!
user_embedding/embedding_lookup?
(user_embedding/embedding_lookup/IdentityIdentity(user_embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*9
_class/
-+loc:@user_embedding/embedding_lookup/201095*+
_output_shapes
:?????????2*
(user_embedding/embedding_lookup/Identity?
*user_embedding/embedding_lookup/Identity_1Identity1user_embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2,
*user_embedding/embedding_lookup/Identity_1?
dot_product/l2_normalize/SquareSquare4movie_embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????2!
dot_product/l2_normalize/Square?
.dot_product/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.dot_product/l2_normalize/Sum/reduction_indices?
dot_product/l2_normalize/SumSum#dot_product/l2_normalize/Square:y:07dot_product/l2_normalize/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2
dot_product/l2_normalize/Sum?
"dot_product/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+2$
"dot_product/l2_normalize/Maximum/y?
 dot_product/l2_normalize/MaximumMaximum%dot_product/l2_normalize/Sum:output:0+dot_product/l2_normalize/Maximum/y:output:0*
T0*+
_output_shapes
:?????????2"
 dot_product/l2_normalize/Maximum?
dot_product/l2_normalize/RsqrtRsqrt$dot_product/l2_normalize/Maximum:z:0*
T0*+
_output_shapes
:?????????2 
dot_product/l2_normalize/Rsqrt?
dot_product/l2_normalizeMul4movie_embedding/embedding_lookup/Identity_1:output:0"dot_product/l2_normalize/Rsqrt:y:0*
T0*+
_output_shapes
:?????????2
dot_product/l2_normalize?
!dot_product/l2_normalize_1/SquareSquare3user_embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:?????????2#
!dot_product/l2_normalize_1/Square?
0dot_product/l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :22
0dot_product/l2_normalize_1/Sum/reduction_indices?
dot_product/l2_normalize_1/SumSum%dot_product/l2_normalize_1/Square:y:09dot_product/l2_normalize_1/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:?????????*
	keep_dims(2 
dot_product/l2_normalize_1/Sum?
$dot_product/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *̼?+2&
$dot_product/l2_normalize_1/Maximum/y?
"dot_product/l2_normalize_1/MaximumMaximum'dot_product/l2_normalize_1/Sum:output:0-dot_product/l2_normalize_1/Maximum/y:output:0*
T0*+
_output_shapes
:?????????2$
"dot_product/l2_normalize_1/Maximum?
 dot_product/l2_normalize_1/RsqrtRsqrt&dot_product/l2_normalize_1/Maximum:z:0*
T0*+
_output_shapes
:?????????2"
 dot_product/l2_normalize_1/Rsqrt?
dot_product/l2_normalize_1Mul3user_embedding/embedding_lookup/Identity_1:output:0$dot_product/l2_normalize_1/Rsqrt:y:0*
T0*+
_output_shapes
:?????????2
dot_product/l2_normalize_1?
dot_product/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
dot_product/transpose/perm?
dot_product/transpose	Transposedot_product/l2_normalize_1:z:0#dot_product/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
dot_product/transpose?
dot_product/MatMulBatchMatMulV2dot_product/l2_normalize:z:0dot_product/transpose:y:0*
T0*+
_output_shapes
:?????????2
dot_product/MatMulq
dot_product/ShapeShapedot_product/MatMul:output:0*
T0*
_output_shapes
:2
dot_product/Shapes
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_1/Const?
flatten_1/ReshapeReshapedot_product/MatMul:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_1/Reshape?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulflatten_1/Reshape:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_5/BiasAdd?
dense_5/re_lu_4/ReluReludense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_5/re_lu_4/Relu?
"dense_5/ActivityRegularizer/SquareSquare"dense_5/re_lu_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2$
"dense_5/ActivityRegularizer/Square?
!dense_5/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_5/ActivityRegularizer/Const?
dense_5/ActivityRegularizer/SumSum&dense_5/ActivityRegularizer/Square:y:0*dense_5/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_5/ActivityRegularizer/Sum?
!dense_5/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_5/ActivityRegularizer/mul/x?
dense_5/ActivityRegularizer/mulMul*dense_5/ActivityRegularizer/mul/x:output:0(dense_5/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_5/ActivityRegularizer/mul?
!dense_5/ActivityRegularizer/ShapeShape"dense_5/re_lu_4/Relu:activations:0*
T0*
_output_shapes
:2#
!dense_5/ActivityRegularizer/Shape?
/dense_5/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_5/ActivityRegularizer/strided_slice/stack?
1dense_5/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_1?
1dense_5/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_5/ActivityRegularizer/strided_slice/stack_2?
)dense_5/ActivityRegularizer/strided_sliceStridedSlice*dense_5/ActivityRegularizer/Shape:output:08dense_5/ActivityRegularizer/strided_slice/stack:output:0:dense_5/ActivityRegularizer/strided_slice/stack_1:output:0:dense_5/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_5/ActivityRegularizer/strided_slice?
 dense_5/ActivityRegularizer/CastCast2dense_5/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_5/ActivityRegularizer/Cast?
#dense_5/ActivityRegularizer/truedivRealDiv#dense_5/ActivityRegularizer/mul:z:0$dense_5/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_5/ActivityRegularizer/truediv?
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_6/MatMul/ReadVariableOp?
dense_6/MatMulMatMul"dense_5/re_lu_4/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/MatMul?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/BiasAdd?
dense_6/re_lu_5/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_6/re_lu_5/Relu?
"dense_6/ActivityRegularizer/SquareSquare"dense_6/re_lu_5/Relu:activations:0*
T0*'
_output_shapes
:?????????2$
"dense_6/ActivityRegularizer/Square?
!dense_6/ActivityRegularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_6/ActivityRegularizer/Const?
dense_6/ActivityRegularizer/SumSum&dense_6/ActivityRegularizer/Square:y:0*dense_6/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_6/ActivityRegularizer/Sum?
!dense_6/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_6/ActivityRegularizer/mul/x?
dense_6/ActivityRegularizer/mulMul*dense_6/ActivityRegularizer/mul/x:output:0(dense_6/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_6/ActivityRegularizer/mul?
!dense_6/ActivityRegularizer/ShapeShape"dense_6/re_lu_5/Relu:activations:0*
T0*
_output_shapes
:2#
!dense_6/ActivityRegularizer/Shape?
/dense_6/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_6/ActivityRegularizer/strided_slice/stack?
1dense_6/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_1?
1dense_6/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_6/ActivityRegularizer/strided_slice/stack_2?
)dense_6/ActivityRegularizer/strided_sliceStridedSlice*dense_6/ActivityRegularizer/Shape:output:08dense_6/ActivityRegularizer/strided_slice/stack:output:0:dense_6/ActivityRegularizer/strided_slice/stack_1:output:0:dense_6/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_6/ActivityRegularizer/strided_slice?
 dense_6/ActivityRegularizer/CastCast2dense_6/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_6/ActivityRegularizer/Cast?
#dense_6/ActivityRegularizer/truedivRealDiv#dense_6/ActivityRegularizer/mul:z:0$dense_6/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_6/ActivityRegularizer/truediv?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_7/MatMul/ReadVariableOp?
dense_7/MatMulMatMul"dense_6/re_lu_5/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/MatMul?
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_7/BiasAddy
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_7/Sigmoid?
IdentityIdentitydense_7/Sigmoid:y:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp!^movie_embedding/embedding_lookup ^user_embedding/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity'dense_5/ActivityRegularizer/truediv:z:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp!^movie_embedding/embedding_lookup ^user_embedding/embedding_lookup*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity'dense_6/ActivityRegularizer/truediv:z:0^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp!^movie_embedding/embedding_lookup ^user_embedding/embedding_lookup*
T0*
_output_shapes
: 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????:?????????: : : : : : : : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2D
 movie_embedding/embedding_lookup movie_embedding/embedding_lookup2B
user_embedding/embedding_lookupuser_embedding/embedding_lookup:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
??
?
"__inference__traced_restore_201582
file_prefix>
+assignvariableop_movie_embedding_embeddings:	??
,assignvariableop_1_user_embedding_embeddings:	?/4
!assignvariableop_2_dense_5_kernel:	?-
assignvariableop_3_dense_5_bias:3
!assignvariableop_4_dense_6_kernel:-
assignvariableop_5_dense_6_bias:3
!assignvariableop_6_dense_7_kernel:-
assignvariableop_7_dense_7_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: #
assignvariableop_13_total: #
assignvariableop_14_count: H
5assignvariableop_15_adam_movie_embedding_embeddings_m:	?G
4assignvariableop_16_adam_user_embedding_embeddings_m:	?/<
)assignvariableop_17_adam_dense_5_kernel_m:	?5
'assignvariableop_18_adam_dense_5_bias_m:;
)assignvariableop_19_adam_dense_6_kernel_m:5
'assignvariableop_20_adam_dense_6_bias_m:;
)assignvariableop_21_adam_dense_7_kernel_m:5
'assignvariableop_22_adam_dense_7_bias_m:H
5assignvariableop_23_adam_movie_embedding_embeddings_v:	?G
4assignvariableop_24_adam_user_embedding_embeddings_v:	?/<
)assignvariableop_25_adam_dense_5_kernel_v:	?5
'assignvariableop_26_adam_dense_5_bias_v:;
)assignvariableop_27_adam_dense_6_kernel_v:5
'assignvariableop_28_adam_dense_6_bias_v:;
)assignvariableop_29_adam_dense_7_kernel_v:5
'assignvariableop_30_adam_dense_7_bias_v:
identity_32??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-1/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::*.
dtypes$
"2 	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp+assignvariableop_movie_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp,assignvariableop_1_user_embedding_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_6_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_6_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_7_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_7_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp5assignvariableop_15_adam_movie_embedding_embeddings_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp4assignvariableop_16_adam_user_embedding_embeddings_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_5_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_5_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_6_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_6_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_7_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_7_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp5assignvariableop_23_adam_movie_embedding_embeddings_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp4assignvariableop_24_adam_user_embedding_embeddings_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_5_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_5_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_6_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_6_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_7_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_7_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_309
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_31?
Identity_32IdentityIdentity_31:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_32"#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
K__inference_movie_embedding_layer_call_and_return_conditional_losses_200435

inputs*
embedding_lookup_200429:	?
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_200429Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0**
_class 
loc:@embedding_lookup/200429*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@embedding_lookup/200429*+
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_200516

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_4_layer_call_fn_201272

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2006502
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_dense_5_layer_call_and_return_conditional_losses_201351

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddh
re_lu_4/ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
re_lu_4/Relu?
IdentityIdentityre_lu_4/Relu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_201277

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
F
/__inference_dense_6_activity_regularizer_200416
x
identity@
SquareSquarex*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
?
F
/__inference_dense_5_activity_regularizer_200403
x
identity@
SquareSquarex*
T0*
_output_shapes
:2
SquareA
RankRank
Square:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:?????????2
rangeN
SumSum
Square:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulJ
IdentityIdentitymul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex
?
?
/__inference_user_embedding_layer_call_fn_201192

inputs
unknown:	?/
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_user_embedding_layer_call_and_return_conditional_losses_2004492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
J__inference_user_embedding_layer_call_and_return_conditional_losses_201202

inputs*
embedding_lookup_201196:	?/
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_201196Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0**
_class 
loc:@embedding_lookup/201196*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@embedding_lookup/201196*+
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookup*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
movie_input4
serving_default_movie_input:0?????????
A

user_input3
serving_default_user_input:0?????????;
dense_70
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?Y
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?U
_tf_keras_network?U{"name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 22]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "movie_input"}, "name": "movie_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 22]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "user_input"}, "name": "user_input", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "movie_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 3706, "output_dim": 4, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "movie_embedding", "inbound_nodes": [[["movie_input", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "user_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 6040, "output_dim": 4, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "user_embedding", "inbound_nodes": [[["user_input", 0, 0, {}]]]}, {"class_name": "Dot", "config": {"name": "dot_product", "trainable": true, "dtype": "float32", "axes": 2, "normalize": true}, "name": "dot_product", "inbound_nodes": [[["movie_embedding", 0, 0, {}], ["user_embedding", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["dot_product", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 8, "activation": {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 8, "activation": {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dropout_4", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_5", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 5, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dropout_5", 0, 0, {}]]]}], "input_layers": [["movie_input", 0, 0], ["user_input", 0, 0]], "output_layers": [["dense_7", 0, 0]]}, "shared_object_id": 23, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 22]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 22]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 22]}, {"class_name": "TensorShape", "items": [null, 22]}], "is_graph_network": true, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 22]}, "float32", "movie_input"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 22]}, "float32", "user_input"]}], "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 22]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "movie_input"}, "name": "movie_input", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 22]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "user_input"}, "name": "user_input", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "Embedding", "config": {"name": "movie_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 3706, "output_dim": 4, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "movie_embedding", "inbound_nodes": [[["movie_input", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Embedding", "config": {"name": "user_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 6040, "output_dim": 4, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 4}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "name": "user_embedding", "inbound_nodes": [[["user_input", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Dot", "config": {"name": "dot_product", "trainable": true, "dtype": "float32", "axes": 2, "normalize": true}, "name": "dot_product", "inbound_nodes": [[["movie_embedding", 0, 0, {}], ["user_embedding", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["dot_product", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 8, "activation": {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "shared_object_id": 8}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.0}, "shared_object_id": 11}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["flatten_1", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["dense_5", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 8, "activation": {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "shared_object_id": 14}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.0}, "shared_object_id": 17}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dropout_4", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_5", "inbound_nodes": [[["dense_6", 0, 0, {}]]], "shared_object_id": 19}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 5, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dropout_5", 0, 0, {}]]], "shared_object_id": 22}], "input_layers": [["movie_input", 0, 0], ["user_input", 0, 0]], "output_layers": [["dense_7", 0, 0]]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 26}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.008019213564693928, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "movie_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 22]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 22]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "movie_input"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "user_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 22]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 22]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "user_input"}}
?

embeddings
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "movie_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "movie_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 3706, "output_dim": 4, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "inbound_nodes": [[["movie_input", 0, 0, {}]]], "shared_object_id": 3, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22]}}
?

embeddings
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "user_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "user_embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 6040, "output_dim": 4, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 4}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "inbound_nodes": [[["user_input", 0, 0, {}]]], "shared_object_id": 5, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22]}}
?
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dot_product", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dot", "config": {"name": "dot_product", "trainable": true, "dtype": "float32", "axes": 2, "normalize": true}, "inbound_nodes": [[["movie_embedding", 0, 0, {}], ["user_embedding", 0, 0, {}]]], "shared_object_id": 6, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 22, 4]}, {"class_name": "TensorShape", "items": [null, 22, 4]}]}
?
 trainable_variables
!regularization_losses
"	variables
#	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["dot_product", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 27}}
?
$
activation

%kernel
&bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 8, "activation": {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "shared_object_id": 8}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.0}, "shared_object_id": 11}, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_1", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 484}}, "shared_object_id": 28}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.0}, "shared_object_id": 11}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 484]}}
?
+trainable_variables
,regularization_losses
-	variables
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_5", 0, 0, {}]]], "shared_object_id": 13}
?
/
activation

0kernel
1bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 8, "activation": {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "shared_object_id": 14}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.0}, "shared_object_id": 17}, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_4", 0, 0, {}]]], "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}, "shared_object_id": 29}, "activity_regularizer": {"class_name": "L2", "config": {"l2": 0.0}, "shared_object_id": 17}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
?
6trainable_variables
7regularization_losses
8	variables
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "inbound_nodes": [[["dense_6", 0, 0, {}]]], "shared_object_id": 19}
?	

:kernel
;bias
<trainable_variables
=regularization_losses
>	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 5, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_5", 0, 0, {}]]], "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
?
@iter

Abeta_1

Bbeta_2
	Cdecay
Dlearning_ratem?m?%m?&m?0m?1m?:m?;m?v?v?%v?&v?0v?1v?:v?;v?"
	optimizer
X
0
1
%2
&3
04
15
:6
;7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
%2
&3
04
15
:6
;7"
trackable_list_wrapper
?
trainable_variables
Elayer_regularization_losses

Flayers
Gnon_trainable_variables
Hmetrics
regularization_losses
	variables
Ilayer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
-:+	?2movie_embedding/embeddings
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
trainable_variables
Jlayer_regularization_losses

Klayers
Lnon_trainable_variables
Mmetrics
regularization_losses
	variables
Nlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*	?/2user_embedding/embeddings
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
trainable_variables
Olayer_regularization_losses

Players
Qnon_trainable_variables
Rmetrics
regularization_losses
	variables
Slayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
Tlayer_regularization_losses

Ulayers
Vnon_trainable_variables
Wmetrics
regularization_losses
	variables
Xlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 trainable_variables
Ylayer_regularization_losses

Zlayers
[non_trainable_variables
\metrics
!regularization_losses
"	variables
]layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
^trainable_variables
_regularization_losses
`	variables
a	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "shared_object_id": 8}
!:	?2dense_5/kernel
:2dense_5/bias
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
?
'trainable_variables
blayer_regularization_losses

clayers
dnon_trainable_variables
emetrics
(regularization_losses
)	variables
flayer_metrics
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
+trainable_variables
glayer_regularization_losses

hlayers
inon_trainable_variables
jmetrics
,regularization_losses
-	variables
klayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
ltrainable_variables
mregularization_losses
n	variables
o	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "shared_object_id": 14}
 :2dense_6/kernel
:2dense_6/bias
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
?
2trainable_variables
player_regularization_losses

qlayers
rnon_trainable_variables
smetrics
3regularization_losses
4	variables
tlayer_metrics
?__call__
?activity_regularizer_fn
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
6trainable_variables
ulayer_regularization_losses

vlayers
wnon_trainable_variables
xmetrics
7regularization_losses
8	variables
ylayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense_7/kernel
:2dense_7/bias
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
?
<trainable_variables
zlayer_regularization_losses

{layers
|non_trainable_variables
}metrics
=regularization_losses
>	variables
~layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
^trainable_variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
_regularization_losses
`	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
$0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
ltrainable_variables
 ?layer_regularization_losses
?layers
?non_trainable_variables
?metrics
mregularization_losses
n	variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
/0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 31}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
2:0	?2!Adam/movie_embedding/embeddings/m
1:/	?/2 Adam/user_embedding/embeddings/m
&:$	?2Adam/dense_5/kernel/m
:2Adam/dense_5/bias/m
%:#2Adam/dense_6/kernel/m
:2Adam/dense_6/bias/m
%:#2Adam/dense_7/kernel/m
:2Adam/dense_7/bias/m
2:0	?2!Adam/movie_embedding/embeddings/v
1:/	?/2 Adam/user_embedding/embeddings/v
&:$	?2Adam/dense_5/kernel/v
:2Adam/dense_5/bias/v
%:#2Adam/dense_6/kernel/v
:2Adam/dense_6/bias/v
%:#2Adam/dense_7/kernel/v
:2Adam/dense_7/bias/v
?2?
(__inference_model_1_layer_call_fn_200591
(__inference_model_1_layer_call_fn_200974
(__inference_model_1_layer_call_fn_200998
(__inference_model_1_layer_call_fn_200824?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_200390?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *U?R
P?M
%?"
movie_input?????????
$?!

user_input?????????
?2?
C__inference_model_1_layer_call_and_return_conditional_losses_201084
C__inference_model_1_layer_call_and_return_conditional_losses_201168
C__inference_model_1_layer_call_and_return_conditional_losses_200872
C__inference_model_1_layer_call_and_return_conditional_losses_200920?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
0__inference_movie_embedding_layer_call_fn_201175?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_movie_embedding_layer_call_and_return_conditional_losses_201185?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_user_embedding_layer_call_fn_201192?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_user_embedding_layer_call_and_return_conditional_losses_201202?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_dot_product_layer_call_fn_201208?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dot_product_layer_call_and_return_conditional_losses_201231?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_1_layer_call_fn_201236?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_1_layer_call_and_return_conditional_losses_201242?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_5_layer_call_fn_201251?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_5_layer_call_and_return_all_conditional_losses_201262?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_4_layer_call_fn_201267
*__inference_dropout_4_layer_call_fn_201272?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_4_layer_call_and_return_conditional_losses_201277
E__inference_dropout_4_layer_call_and_return_conditional_losses_201281?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_6_layer_call_fn_201290?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_dense_6_layer_call_and_return_all_conditional_losses_201301?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_5_layer_call_fn_201306
*__inference_dropout_5_layer_call_fn_201311?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_5_layer_call_and_return_conditional_losses_201316
E__inference_dropout_5_layer_call_and_return_conditional_losses_201320?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_7_layer_call_fn_201329?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_7_layer_call_and_return_conditional_losses_201340?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_200950movie_input
user_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_dense_5_activity_regularizer_200403?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
C__inference_dense_5_layer_call_and_return_conditional_losses_201351?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_dense_6_activity_regularizer_200416?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
?2?
C__inference_dense_6_layer_call_and_return_conditional_losses_201362?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_200390?%&01:;_?\
U?R
P?M
%?"
movie_input?????????
$?!

user_input?????????
? "1?.
,
dense_7!?
dense_7?????????Y
/__inference_dense_5_activity_regularizer_200403&?
?
?	
x
? "? ?
G__inference_dense_5_layer_call_and_return_all_conditional_losses_201262k%&0?-
&?#
!?
inputs??????????
? "3?0
?
0?????????
?
?	
1/0 ?
C__inference_dense_5_layer_call_and_return_conditional_losses_201351]%&0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_dense_5_layer_call_fn_201251P%&0?-
&?#
!?
inputs??????????
? "??????????Y
/__inference_dense_6_activity_regularizer_200416&?
?
?	
x
? "? ?
G__inference_dense_6_layer_call_and_return_all_conditional_losses_201301j01/?,
%?"
 ?
inputs?????????
? "3?0
?
0?????????
?
?	
1/0 ?
C__inference_dense_6_layer_call_and_return_conditional_losses_201362\01/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_6_layer_call_fn_201290O01/?,
%?"
 ?
inputs?????????
? "???????????
C__inference_dense_7_layer_call_and_return_conditional_losses_201340\:;/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_7_layer_call_fn_201329O:;/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_dot_product_layer_call_and_return_conditional_losses_201231?b?_
X?U
S?P
&?#
inputs/0?????????
&?#
inputs/1?????????
? ")?&
?
0?????????
? ?
,__inference_dot_product_layer_call_fn_201208?b?_
X?U
S?P
&?#
inputs/0?????????
&?#
inputs/1?????????
? "???????????
E__inference_dropout_4_layer_call_and_return_conditional_losses_201277\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
E__inference_dropout_4_layer_call_and_return_conditional_losses_201281\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? }
*__inference_dropout_4_layer_call_fn_201267O3?0
)?&
 ?
inputs?????????
p 
? "??????????}
*__inference_dropout_4_layer_call_fn_201272O3?0
)?&
 ?
inputs?????????
p
? "???????????
E__inference_dropout_5_layer_call_and_return_conditional_losses_201316\3?0
)?&
 ?
inputs?????????
p 
? "%?"
?
0?????????
? ?
E__inference_dropout_5_layer_call_and_return_conditional_losses_201320\3?0
)?&
 ?
inputs?????????
p
? "%?"
?
0?????????
? }
*__inference_dropout_5_layer_call_fn_201306O3?0
)?&
 ?
inputs?????????
p 
? "??????????}
*__inference_dropout_5_layer_call_fn_201311O3?0
)?&
 ?
inputs?????????
p
? "???????????
E__inference_flatten_1_layer_call_and_return_conditional_losses_201242]3?0
)?&
$?!
inputs?????????
? "&?#
?
0??????????
? ~
*__inference_flatten_1_layer_call_fn_201236P3?0
)?&
$?!
inputs?????????
? "????????????
C__inference_model_1_layer_call_and_return_conditional_losses_200872?%&01:;g?d
]?Z
P?M
%?"
movie_input?????????
$?!

user_input?????????
p 

 
? "A?>
?
0?????????
?
?	
1/0 
?	
1/1 ?
C__inference_model_1_layer_call_and_return_conditional_losses_200920?%&01:;g?d
]?Z
P?M
%?"
movie_input?????????
$?!

user_input?????????
p

 
? "A?>
?
0?????????
?
?	
1/0 
?	
1/1 ?
C__inference_model_1_layer_call_and_return_conditional_losses_201084?%&01:;b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "A?>
?
0?????????
?
?	
1/0 
?	
1/1 ?
C__inference_model_1_layer_call_and_return_conditional_losses_201168?%&01:;b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "A?>
?
0?????????
?
?	
1/0 
?	
1/1 ?
(__inference_model_1_layer_call_fn_200591?%&01:;g?d
]?Z
P?M
%?"
movie_input?????????
$?!

user_input?????????
p 

 
? "???????????
(__inference_model_1_layer_call_fn_200824?%&01:;g?d
]?Z
P?M
%?"
movie_input?????????
$?!

user_input?????????
p

 
? "???????????
(__inference_model_1_layer_call_fn_200974?%&01:;b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "???????????
(__inference_model_1_layer_call_fn_200998?%&01:;b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "???????????
K__inference_movie_embedding_layer_call_and_return_conditional_losses_201185_/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? ?
0__inference_movie_embedding_layer_call_fn_201175R/?,
%?"
 ?
inputs?????????
? "???????????
$__inference_signature_wrapper_200950?%&01:;w?t
? 
m?j
4
movie_input%?"
movie_input?????????
2

user_input$?!

user_input?????????"1?.
,
dense_7!?
dense_7??????????
J__inference_user_embedding_layer_call_and_return_conditional_losses_201202_/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? ?
/__inference_user_embedding_layer_call_fn_201192R/?,
%?"
 ?
inputs?????????
? "??????????