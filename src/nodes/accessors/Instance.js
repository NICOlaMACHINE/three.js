
import { vec3, mat4, Fn } from '../tsl/TSLBase.js';
import { OnAfterObjectUpdate, OnBeforeFrameUpdate, OnBeforeObjectUpdate } from '../utils/EventNode.js';
import { normalLocal, transformNormal } from './Normal.js';
import { positionLocal, positionPrevious } from './Position.js';
import { varyingProperty } from '../core/PropertyNode.js';
import { instancedBufferAttribute, instancedDynamicBufferAttribute } from './BufferAttributeNode.js';
import { buffer } from './BufferNode.js';
import { storage } from './StorageBufferNode.js';
import { instanceIndex } from '../core/IndexNode.js';

import { InstancedInterleavedBuffer } from '../../core/InstancedInterleavedBuffer.js';
import { InstancedBufferAttribute } from '../../core/InstancedBufferAttribute.js';
import { InterleavedBufferAttribute } from '../../core/InterleavedBufferAttribute.js';
import { getDataFromObject } from '../core/NodeUtils.js';
import { DynamicDrawUsage } from '../../constants.js';

const _matrixBuffers = /*@__PURE__*/ new WeakMap();
const _colorBuffers = /*@__PURE__*/ new WeakMap();
const _previousInstanceMatrices = /*@__PURE__*/ new WeakMap();

/**
 * Returns `true` if the instanced mesh can share its node builder state with
 * other instanced meshes. Shared programs read the instance matrices of the
 * object being rendered instead of embedding the buffers of a specific mesh.
 * Storage buffers, instance colors and previous-frame data for motion vectors
 * remain per object, as does instancing outside WebGPURenderer.
 *
 * @param {InstancedMesh} object - The instanced mesh.
 * @param {Renderer} renderer - The renderer.
 * @returns {boolean} Whether the instancing setup can be shared.
 */
export function isSharedInstancing( object, renderer ) {

	// Only WebGPURenderer's render objects share node builder states; other
	// renderers (e.g. WebGLRenderer with node materials) keep the per-object setup.
	if ( renderer.isWebGPURenderer !== true ) return false;

	if ( object.isInstancedMesh !== true || object.instanceColor !== null ) return false;

	const instanceMatrix = object.instanceMatrix;

	if ( ! instanceMatrix || instanceMatrix.isInstancedBufferAttribute !== true || instanceMatrix.isStorageInstancedBufferAttribute === true ) return false;

	const mrt = renderer.getMRT();

	return ! ( mrt && mrt.has( 'velocity' ) ) && getDataFromObject( object ).useVelocity !== true;

}

/**
 * Returns the interleaved buffer that feeds the instance matrix attributes.
 *
 * @param {InstancedBufferAttribute} instanceMatrix - The matrix buffer attribute.
 * @returns {InstancedInterleavedBuffer} The interleaved buffer.
 */
function getInterleavedMatrix( instanceMatrix ) {

	let interleaved = _matrixBuffers.get( instanceMatrix );

	if ( ! interleaved ) {

		interleaved = new InstancedInterleavedBuffer( instanceMatrix.array, 16, 1 );
		_matrixBuffers.set( instanceMatrix, interleaved );

	}

	return interleaved;

}

/**
 * Copies pending matrix updates into the interleaved buffer used for rendering.
 *
 * @param {InstancedBufferAttribute} matrices - The source matrix attribute.
 * @param {?InstancedInterleavedBuffer} interleavedMatrix - The interleaved buffer.
 */
function syncInterleavedMatrix( matrices, interleavedMatrix ) {

	if ( interleavedMatrix !== null && interleavedMatrix.version !== matrices.version ) {

		interleavedMatrix.clearUpdateRanges();
		interleavedMatrix.updateRanges.push( ...matrices.updateRanges );
		matrices.clearUpdateRanges(); // "matrices" as the source is never uploaded directly. clear to avoid update range accumulation

		interleavedMatrix.version = matrices.version;

	}

}

/**
 * Creates the instance matrix node of a shared instancing setup. The matrices
 * are always read from instanced attributes, which each render object binds
 * to its own buffer, so the program does not depend on the instance count.
 *
 * @param {InstancedBufferAttribute} instanceMatrix - The matrix buffer attribute of the object being built.
 * @returns {Node} The matrix node.
 */
function createSharedInstanceMatrixNode( instanceMatrix ) {

	const interleaved = getInterleavedMatrix( instanceMatrix );

	const columns = [ 0, 4, 8, 12 ].map( offset => {

		const node = instancedBufferAttribute( interleaved, 'vec4', 16, offset );

		// Each render object binds the matrices of its own mesh.
		node.getObjectAttribute = object => {

			const buffer = getInterleavedMatrix( object.instanceMatrix ).setUsage( object.instanceMatrix.usage );
			const attribute = new InterleavedBufferAttribute( buffer, 4, offset );
			attribute.isInstancedBufferAttribute = true;

			return attribute;

		};

		return node;

	} );

	OnBeforeObjectUpdate( ( { object } ) => {

		syncInterleavedMatrix( object.instanceMatrix, getInterleavedMatrix( object.instanceMatrix ) );

	} );

	return mat4( ...columns );

}

/**
 * Creates the appropriate node for instanced matrix transformations.
 * Depending on buffer limits and storage capability, returns either a storage, buffer, or instanced interleaved attribute node.
 *
 * @param {NodeBuilder} builder - The current node builder.
 * @param {InstancedBufferAttribute|StorageInstancedBufferAttribute} instanceMatrix - The matrix buffer attribute.
 * @returns {Node} The matrix node.
 */
function createInstanceMatrixNode( builder, instanceMatrix ) {

	let instanceMatrixNode;
	const matrixCount = Math.max( instanceMatrix.count, 1 );

	const isStorageMatrix = instanceMatrix.isStorageInstancedBufferAttribute === true;

	if ( isStorageMatrix ) {

		instanceMatrixNode = storage( instanceMatrix, 'mat4', matrixCount ).element( instanceIndex );

	} else {

		const uniformBufferSize = matrixCount * 16 * 4;

		if ( uniformBufferSize <= builder.getUniformBufferLimit() ) {

			instanceMatrixNode = buffer( instanceMatrix.array, 'mat4', matrixCount ).element( instanceIndex );

		} else {

			const interleaved = getInterleavedMatrix( instanceMatrix );

			const bufferFn = instanceMatrix.usage === DynamicDrawUsage ? instancedDynamicBufferAttribute : instancedBufferAttribute;

			const instanceBuffers = [
				bufferFn( interleaved, 'vec4', 16, 0 ),
				bufferFn( interleaved, 'vec4', 16, 4 ),
				bufferFn( interleaved, 'vec4', 16, 8 ),
				bufferFn( interleaved, 'vec4', 16, 12 )
			];

			instanceMatrixNode = mat4( ...instanceBuffers );

		}

	}

	return instanceMatrixNode;

}

/**
 * Retrieves or initializes the previous frame instance matrix node for motion vectors.
 * Uses a WeakMap to cache previous frame instance matrices and their TSL nodes.
 *
 * @param {InstancedMesh} instancedMesh - The instanced mesh object.
 * @param {InstancedBufferAttribute|StorageInstancedBufferAttribute} instanceMatrix - The current matrix buffer attribute.
 * @param {NodeBuilder} builder - The current node builder.
 * @returns {Node} The previous frame instance matrix node.
 */
function getPreviousInstance( instancedMesh, instanceMatrix, builder ) {

	let data = _previousInstanceMatrices.get( instancedMesh );

	if ( data === undefined ) {

		const previousInstanceMatrix = instanceMatrix.clone();

		data = {
			previousInstanceMatrix,
			node: createInstanceMatrixNode( builder, previousInstanceMatrix )
		};

		_previousInstanceMatrices.set( instancedMesh, data );

	}

	return data.node;

}

/**
 * TSL object representing a varying property for the instanced color vector.
 *
 * @type {VaryingNode<vec3>}
 */
export const instanceColor = /*@__PURE__*/ varyingProperty( 'vec3', 'vInstanceColor' );

/**
 * TSL function representing the standard instancing vertex shader setup.
 * Transforms positionLocal and normalLocal, and assigns varying color in-place.
 *
 * @tsl
 * @function
 * @param {InstancedBufferAttribute|StorageInstancedBufferAttribute} matrices - The instanced transformation matrices.
 * @param {?InstancedBufferAttribute|StorageInstancedBufferAttribute} [colors=null] - The optional instanced colors.
 */
export const instance = /*@__PURE__*/ Fn( ( [ matrices, colors = null ], builder ) => {

	setupInstance( builder, matrices, colors, false );

}, 'void' );

/**
 * Sets up instanced transformations and colors.
 *
 * @private
 * @param {NodeBuilder} builder - The current node builder.
 * @param {InstancedBufferAttribute|StorageInstancedBufferAttribute} matrices - The instanced transformation matrices.
 * @param {?InstancedBufferAttribute|StorageInstancedBufferAttribute} colors - The optional instanced colors.
 * @param {boolean} shared - Whether the matrices are resolved from the rendered object, so the program can be shared.
 */
function setupInstance( builder, matrices, colors, shared ) {

	const isStorageMatrix = matrices.isStorageInstancedBufferAttribute === true;
	const isStorageColor = colors && colors.isStorageInstancedBufferAttribute === true;

	const instanceMatrixNode = shared ? createSharedInstanceMatrixNode( matrices ) : createInstanceMatrixNode( builder, matrices );

	// interleaved buffer tracking for matrix
	let interleavedMatrix = null;

	if ( ! isStorageMatrix && ! shared ) {

		const uniformBufferSize = Math.max( matrices.count, 1 ) * 16 * 4;

		if ( uniformBufferSize > builder.getUniformBufferLimit() ) {

			interleavedMatrix = _matrixBuffers.get( matrices );

		}

	}

	let instanceColorNode = null;
	let interleavedColor = null;

	if ( colors ) {

		if ( isStorageColor ) {

			instanceColorNode = storage( colors, 'vec3', Math.max( colors.count, 1 ) ).element( instanceIndex );

		} else {

			let bufferAttribute = _colorBuffers.get( colors );

			if ( ! bufferAttribute ) {

				bufferAttribute = new InstancedBufferAttribute( colors.array, 3 );
				_colorBuffers.set( colors, bufferAttribute );

			}

			interleavedColor = bufferAttribute;

			const bufferFn = colors.usage === DynamicDrawUsage ? instancedDynamicBufferAttribute : instancedBufferAttribute;

			instanceColorNode = vec3( bufferFn( bufferAttribute, 'vec3', 3, 0 ) );

		}

	}

	// Synchronization of dynamic buffer updates per frame.
	if ( interleavedMatrix !== null || interleavedColor !== null ) {

		OnBeforeFrameUpdate( () => {

			syncInterleavedMatrix( matrices, interleavedMatrix );

			if ( colors && interleavedColor !== null && interleavedColor.version !== colors.version ) {

				interleavedColor.clearUpdateRanges();
				interleavedColor.updateRanges.push( ...colors.updateRanges );
				colors.clearUpdateRanges();

				interleavedColor.version = colors.version;

			}

		} );

	}

	// POSITION

	const instancePosition = instanceMatrixNode.mul( positionLocal ).xyz;
	positionLocal.assign( instancePosition );

	if ( builder.needsPreviousData() ) {

		const instancedMesh = builder.object;

		OnAfterObjectUpdate( ( { object } ) => {

			const { previousInstanceMatrix } = _previousInstanceMatrices.get( object );

			previousInstanceMatrix.array.set( matrices.array );
			previousInstanceMatrix.version = matrices.version;

			// handle interleaved path

			const previousInterleavedMatrix = _matrixBuffers.get( previousInstanceMatrix );

			if ( previousInterleavedMatrix !== undefined ) previousInterleavedMatrix.version = matrices.version;

		} );

		const previousInstanceMatrixNode = getPreviousInstance( instancedMesh, matrices, builder );
		positionPrevious.assign( previousInstanceMatrixNode.mul( positionPrevious ).xyz );

	}

	// NORMAL

	if ( builder.hasGeometryAttribute( 'normal' ) ) {

		const instanceNormal = transformNormal( normalLocal, instanceMatrixNode );
		normalLocal.assign( instanceNormal );

	}

	// COLOR

	if ( instanceColorNode !== null ) {

		instanceColor.assign( instanceColorNode );

	}

}

/**
 * TSL wrapper for applying instanced mesh rendering setup.
 *
 * @tsl
 * @function
 * @param {InstancedMesh} instancedMesh - The instanced mesh.
 */
export const instancedMesh = /*@__PURE__*/ Fn( ( [ instancedMesh ], builder ) => {

	const { instanceMatrix, instanceColor } = instancedMesh;

	setupInstance( builder, instanceMatrix, instanceColor, isSharedInstancing( instancedMesh, builder.renderer ) );

}, 'void' );
