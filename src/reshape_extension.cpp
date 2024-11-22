#define DUCKDB_EXTENSION_MAIN

#include "reshape_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>

namespace duckdb {

struct ReshapeScalarFunctionData : public FunctionData {
  explicit ReshapeScalarFunctionData(std::vector<int> shape_p,
                                     std::vector<int> cum_prod_p,
                                     LogicalType internal_type_p,
                                     int outer_length_p, int num_elements_p)
      : shape(std::move(shape_p)), cumulative_product(std::move(cum_prod_p)),
        internal_type(internal_type_p), outer_length(outer_length_p),
        num_elements(num_elements_p) {}

  bool Equals(const FunctionData &other) const override {
    DynamicCastCheck<ReshapeScalarFunctionData>(this);
    const auto *left =
        reinterpret_cast<const ReshapeScalarFunctionData *>(&other);
    const auto *right =
        reinterpret_cast<const ReshapeScalarFunctionData *>(this);

    return left->shape == right->shape &&
           left->cumulative_product == right->cumulative_product &&
           left->internal_type == right->internal_type &&
           left->outer_length == right->outer_length &&
           left->num_elements == right->num_elements;
  }

  unique_ptr<FunctionData> Copy() const override {
    return make_uniq<ReshapeScalarFunctionData>(*this);
  }

  std::vector<int> shape;
  std::vector<int> cumulative_product;
  LogicalType internal_type;
  int outer_length;
  int num_elements;
};

unique_ptr<FunctionData>
ReshapeScalarFunctionBind(ClientContext &context,
                          ScalarFunction &bound_function,
                          vector<unique_ptr<Expression>> &arguments) {
  if (!(arguments.size() == 2 || arguments.size() == 3)) {
    throw BinderException(
        "reshape scalar function bind requires two or three arguments");
  }

  // Make sure first argument is a parameter
  if (arguments[1]->HasParameter()) {
    throw BinderException(
        "First argument to reshape scalar function must be a parameter");
  }
  // Evaluate the expression
  auto shape_val = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);

  // Make sure it's a list
  if (shape_val.type().id() != LogicalTypeId::LIST) {
    throw BinderException(
        "First argument to reshape scalar function must be a list");
  }

  const auto &shape_value = ListValue::GetChildren(shape_val);
  // Make sure it's a list of integers
  for (const auto &element : shape_value) {
    if (element.type().id() != LogicalTypeId::INTEGER) {
      throw BinderException("Shape must be a list of integers");
    }
  }

  std::vector<int> shape;
  shape.reserve(shape_value.size());
  std::transform(shape_value.begin(), shape_value.end(),
                 std::back_inserter(shape),
                 [](const Value &v) { return v.GetValue<int>(); });

  // Make sure it's a list of length > 1
  if (shape.empty()) {
    throw BinderException("Shape must have at least one element");
  }

  // Determine the expected length
  // Construct a cumulative product
  std::vector<int> cumulative_product(shape.size());
  int running_product = 1;
  for (int i = shape.size() - 1; i >= 0; i--) {
    cumulative_product[i] = running_product;
    running_product *= shape[i];
  }

  // If there is a 3rd argument it must be a integer with the length of the
  // input of type list
  int list_length = -1;
  if (arguments.size() == 3) {
    if (arguments[2]->HasParameter()) {
      throw BinderException(
          "Third argument to reshape scalar function must be a parameter");
    }
    auto supports_lists_val =
        ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
    if (supports_lists_val.type().id() != LogicalTypeId::INTEGER) {
      throw BinderException(
          "Third argument to reshape scalar function must be an integer");
    }
    list_length = supports_lists_val.GetValue<int>();
    if (list_length <= 0) {
      throw BinderException("Third argument to reshape scalar function must be "
                            "a positive integer");
    }
  }

  // Determine the corresponding output type, input must be either a single
  // array or a single list
  const auto &input_type = arguments[0]->return_type;

  // Check that its an array
  if (!(input_type.id() == LogicalTypeId::ARRAY ||
        (list_length != -1 && input_type.id() == LogicalTypeId::LIST))) {
    throw BinderException(
        "Input must be an array or list with additional flag provided");
  }
  if (list_length != -1 && input_type.id() != LogicalTypeId::LIST) {
    throw BinderException("Input must be a list with additional flag provided");
  }

  LogicalType child_type;
  if (input_type.id() == LogicalTypeId::LIST) {
    child_type = ListType::GetChildType(input_type);
  } else {
    child_type = ArrayType::GetChildType(input_type);
  }

  auto innermost_type = child_type;
  for (int i = shape.size() - 1; i >= 0; i--) {
    child_type = LogicalType::ARRAY(child_type, shape[i]);
  }
  bound_function.return_type = child_type;
  bound_function.arguments[0] = input_type;

  if (input_type.id() == LogicalTypeId::ARRAY) {
    list_length = ArrayType::GetSize(input_type);
    if (ArrayType::GetSize(input_type) == 0 ||
        ArrayType::GetSize(input_type) != running_product) {
      throw BinderException(
          "Shape values must be >= 1 and input size must be a "
          "multiple of the product of the shape values");
    }
  } else {
    if (list_length != running_product) {
      throw BinderException(
          "Shape values must be >= 1 and input size must be a "
          "multiple of the product of the shape values");
    }
  }

  return make_uniq<ReshapeScalarFunctionData>(
      std::move(shape), std::move(cumulative_product), innermost_type,
      list_length, running_product);
}

template <typename T>
static void ReshapeScalarFun(DataChunk &args, ExpressionState &state,
                             Vector &result) {
  auto &array_vector = args.data[0];

  auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
  const auto &info = func_expr.bind_info->Cast<ReshapeScalarFunctionData>();

  // Convert to unified format
  UnifiedVectorFormat vdata;
  array_vector.ToUnifiedFormat(args.size(), vdata);

  bool is_list = array_vector.GetType().id() == LogicalTypeId::LIST;

  // Check that the number of elements is correct for each row in the input
  if (is_list) {
    const auto *entries = UnifiedVectorFormat::GetData<list_entry_t>(vdata);
    const auto &validity = vdata.validity;
    const auto &sel = *vdata.sel;

    for (idx_t i = 0; i < args.size(); i++) {
      auto idx = sel.get_index(i);
      D_ASSERT(!validity.RowIsValid(idx) ||
               entries[idx].length == info.outer_length);
    }
  }

  // Get the inner child vector
  UnifiedVectorFormat child_data;
  if (is_list) {
    // Make sure that the number of elements is correct
    auto &child_vector = ListVector::GetEntry(array_vector);
    child_vector.ToUnifiedFormat(args.size(), child_data);
  } else {
    auto &child_vector = ArrayVector::GetEntry(array_vector);
    child_vector.ToUnifiedFormat(args.size(), child_data);
  }

  // Go into the inner most array
  Vector *result_output = &ArrayVector::GetEntry(result);
  for (idx_t i = 1; i < info.shape.size(); i++) {
    result_output = &ArrayVector::GetEntry(*result_output);
  }

  const auto *child_data_ptr = UnifiedVectorFormat::GetData<T>(child_data);
  const auto &child_validity = child_data.validity;
  const auto &child_sel = *child_data.sel;
  const auto &input_row_validity = vdata.validity;
  const auto &input_row_sel = *vdata.sel;

  auto *result_data = FlatVector::GetData<T>(*result_output);
  auto &result_validity = FlatVector::Validity(*result_output);

  auto& top_level_res_validity = FlatVector::Validity(result);

  for (idx_t row_idx = 0; row_idx < args.size(); row_idx++) {
    // Use vdata to check row validity
    auto input_row_idx = input_row_sel.get_index(row_idx);
    if (!input_row_validity.RowIsValid(input_row_idx)) {
      FlatVector::SetNull(result, row_idx, true);
      continue;
    }

    top_level_res_validity.SetValid(row_idx);

    auto element_start_idx = row_idx * info.num_elements;
    for (idx_t i = element_start_idx; i < element_start_idx + info.num_elements; i++) {
      auto child_idx = child_sel.get_index(i);
      if (child_validity.RowIsValid(child_idx)) {
        result_data[i] = child_data_ptr[child_idx];
        result_validity.SetValid(i);
      } else {
        result_validity.SetInvalid(i);
      }
    }
  }

  // if input is a single element set the output to be a constant vector
  if (args.size() == 1) {
    result.SetVectorType(VectorType::CONSTANT_VECTOR);
  }

  result.Verify(args.size());
}

// Wrapper function to dispatch to appropriate template
static void ReshapeScalarFunDispatch(DataChunk &args, ExpressionState &state,
                             Vector &result) {
  Vector* input_child;
  if (args.data[0].GetType().id() == LogicalTypeId::LIST) {
    input_child = &ListVector::GetEntry(args.data[0]);
  } else {
    input_child = &ArrayVector::GetEntry(args.data[0]);
  }
  switch (input_child->GetType().InternalType()) {
    case PhysicalType::BOOL:
    case PhysicalType::INT8:
      ReshapeScalarFun<int8_t>(args, state, result);
      break;
    case PhysicalType::INT16:
      ReshapeScalarFun<int16_t>(args, state, result);
      break;
    case PhysicalType::INT32:
      ReshapeScalarFun<int32_t>(args, state, result);
      break;
    case PhysicalType::INT64:
      ReshapeScalarFun<int64_t>(args, state, result);
      break;
    case PhysicalType::INT128:
      ReshapeScalarFun<hugeint_t>(args, state, result);
      break;
    case PhysicalType::FLOAT:
      ReshapeScalarFun<float>(args, state, result);
      break;
    case PhysicalType::DOUBLE:
      ReshapeScalarFun<double>(args, state, result);
      break;
    case PhysicalType::INTERVAL:
      ReshapeScalarFun<interval_t>(args, state, result);
      break;
    case PhysicalType::UINT8:
      ReshapeScalarFun<uint8_t>(args, state, result);
      break;
    case PhysicalType::UINT16:
      ReshapeScalarFun<uint16_t>(args, state, result);
      break;
    case PhysicalType::UINT32:
      ReshapeScalarFun<uint32_t>(args, state, result);
      break;
    case PhysicalType::UINT64:
      ReshapeScalarFun<uint64_t>(args, state, result);
      break;
    case PhysicalType::UINT128:
      ReshapeScalarFun<uhugeint_t>(args, state, result);
      break;
    default:
      throw NotImplementedException("Unimplemented type for reshape scalar function");
  }
}



static void LoadInternal(DatabaseInstance &instance) {
  auto scalar_func = ScalarFunction(
      "reshape", {LogicalType::ANY, LogicalType::LIST(LogicalType::INTEGER)},
      LogicalType::ANY, ReshapeScalarFunDispatch, ReshapeScalarFunctionBind);
  scalar_func.varargs = LogicalType::INTEGER;
  ExtensionUtil::RegisterFunction(instance, scalar_func);
}

void ReshapeExtension::Load(DuckDB &db) { LoadInternal(*db.instance); }
std::string ReshapeExtension::Name() { return "reshape"; }

std::string ReshapeExtension::Version() const {
#ifdef EXT_VERSION_RESHAPE
  return EXT_VERSION_RESHAPE;
#else
  return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void reshape_init(duckdb::DatabaseInstance &db) {
  duckdb::DuckDB db_wrapper(db);
  db_wrapper.LoadExtension<duckdb::ReshapeExtension>();
}

DUCKDB_EXTENSION_API const char *reshape_version() {
  return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
