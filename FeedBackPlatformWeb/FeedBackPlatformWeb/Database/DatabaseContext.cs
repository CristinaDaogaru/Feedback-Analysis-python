using FeedBackPlatformWeb.Models;
using System;
using System.Collections.Generic;
using System.Data.Entity;
using System.Linq;
using System.Web;

namespace FeedBackPlatformWeb.Database
{
    public class DatabaseContext : DbContext
    {
        public DbSet<Survey> Surveys { get; set; }
        public DbSet<Category> Categories { get; set; }
        public DbSet<Question> Questions { get; set; }
        public DbSet<ClientProfile> ClientProfiles { get; set; }
        public DbSet<Response> Responses { get; set; }

        protected override void OnModelCreating(DbModelBuilder modelBuilder)
        {
            // configures one-to-many relationship
            modelBuilder.Entity<Category>()
                .HasMany<Survey>(c => c.Surveys)
                .WithRequired(s => s.Category)
                .HasForeignKey<int>(s => s.CategoryId);

            modelBuilder.Entity<ClientProfile>()
                .HasMany<Survey>(c => c.Surveys)
                .WithRequired(s => s.Client)
                .HasForeignKey<int>(s => s.ClientId);
        }
    }
}
