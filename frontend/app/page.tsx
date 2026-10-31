import Link from "next/link";
import { Camera, UserPlus } from "lucide-react";

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-950 flex flex-col items-center justify-center p-8">
      <div className="mb-10 text-center">
        <h1 className="text-4xl font-bold text-white mb-2">🎭 Face Attendance</h1>
        <p className="text-gray-400">Hệ thống điểm danh thông minh bằng nhận diện khuôn mặt</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-2xl">
        <Link href="/attendance"
          className="flex flex-col items-center gap-4 p-10 rounded-2xl border-2 border-blue-600
                     bg-blue-950/40 hover:bg-blue-900/40 transition-all hover:scale-105 cursor-pointer">
          <Camera className="w-16 h-16 text-blue-400" />
          <div className="text-center">
            <h2 className="text-2xl font-bold text-white">Điểm Danh</h2>
            <p className="text-blue-300 text-sm mt-1">Nhận diện qua webcam thời gian thực</p>
          </div>
        </Link>

        <Link href="/register"
          className="flex flex-col items-center gap-4 p-10 rounded-2xl border-2 border-purple-600
                     bg-purple-950/40 hover:bg-purple-900/40 transition-all hover:scale-105 cursor-pointer">
          <UserPlus className="w-16 h-16 text-purple-400" />
          <div className="text-center">
            <h2 className="text-2xl font-bold text-white">Đăng Ký</h2>
            <p className="text-purple-300 text-sm mt-1">Thêm nhân viên mới vào hệ thống</p>
          </div>
        </Link>
      </div>
    </main>
  );
}
