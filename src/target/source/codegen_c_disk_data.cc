#include "codegen_c_disk_data.h"

#include "codegen_c.h"

namespace tvm {
namespace codegen {

void DefaultVisitorState::VisitStmt_(const LetStmtNode* op, CodeGenC* codegen) {
  auto& stream = codegen->stream;
  auto& var_idmap_ = codegen->var_idmap_;
  auto& handle_data_type_ = codegen->handle_data_type_;
  std::string value = codegen->PrintExpr(op->value);
  if (codegen->print_ssa_form_) {
    ICHECK(!var_idmap_.count(op->var.get()));
    codegen->var_idmap_[op->var.get()] = value;
  } else {
    codegen->PrintIndent();
    if (op->var.dtype() == DataType::Handle() && handle_data_type_.count(op->var.get())) {
      codegen->PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "* " << codegen->AllocVarID(op->var.get()) << " = (";
      codegen->PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "*)" << value << ";\n";
    } else {
      codegen->PrintType(op->var.dtype(), stream);
      stream << ' ' << codegen->AllocVarID(op->var.get()) << " = " << value << ";\n";
    }
  }
  codegen->PrintStmt(op->body);
}

void MainFuncVisitorState::VisitStmt_(const LetStmtNode* op, CodeGenC* codegen) {
  auto& stream = codegen->stream;
  auto& var_idmap_ = codegen->var_idmap_;
  auto& handle_data_type_ = codegen->handle_data_type_;
  std::string value = codegen->PrintExpr(op->value);
  std::string var_name = codegen->AllocVarID(op->var.get());

  if (codegen->print_ssa_form_) {
    ICHECK(!var_idmap_.count(op->var.get()));
    codegen->var_idmap_[op->var.get()] = value;
  } else {
    codegen->PrintIndent();
    if (op->var.dtype() == DataType::Handle() && handle_data_type_.count(op->var.get())) {
      codegen->PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "* " << codegen->AllocVarID(op->var.get()) << " = (";
      codegen->PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "*)" << value << ";\n";
    } else if (var_name.substr(0, 3).compare("sid") == 0) {
      // SID is workspace variables allocated by TVM
      // we don't load them from disk
      // codegen->PrintIndent();
      if (op->var.dtype() == DataType::Handle() && handle_data_type_.count(op->var.get())) {
        codegen->PrintType(handle_data_type_.at(op->var.get()), stream);
        stream << "* " << var_name << " = (";
        codegen->PrintType(handle_data_type_.at(op->var.get()), stream);
        stream << "*)" << value << ";\n";
      } else {
        codegen->PrintType(op->var.dtype(), stream);
        stream << ' ' << var_name << " = " << value << ";\n";
      }
    } else {
      auto op_call = op->value.as<CallNode>();
      // auto op_var = op->var.get();

      if (auto opt_call_op = op_call->op.as<Op>()) {
        if (op_call->op.same_as(builtin::address_of())) {
          const BufferLoadNode* load = op_call->args[0].as<BufferLoadNode>();
          // int64_t num_elements = 1;

          Array<PrimExpr> shape;
          Array<PrimExpr> shape2 = load->buffer->shape;

          ICHECK(op_call->args.size() == 1 && load);
          ICHECK_EQ(load->indices.size(), 1) << "CodeGenC only supports flat memory allocations.";

          stream << "static float";
          stream << "* " << var_name << "; // Declare without immediate initialization \n";

          // Open file to read binary data
          codegen->PrintIndent();
          stream << "read_file_into_memory(\"rom://" << var_name << ".dat\", &" << var_name
                 << ");\n\n";
        }
      }
    }
  }
  codegen->PrintStmt(op->body);
}

}  // namespace codegen
}  // namespace tvm
